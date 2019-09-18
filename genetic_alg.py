from functools import reduce
from operator import add
import random
import c2p_segmentation
import torch
import torch.distributions as tdist
import loss
import numpy as np
import os
from xyz2grid import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

root_path = './cnnseg/velodyne64/'
pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
protofile = root_path + 'deploy.prototxt'
weightfile = root_path + 'deploy.caffemodel'
pytorchModels = c2p_segmentation.generatePytorch(protofile, weightfile)
dataIdx = 101
pclfile = '%06d.bin'%(dataIdx)
PCL_path = pclpath + pclfile
_, PCL_except_car,target_obs = c2p_segmentation.preProcess(PCL_path,'./101/37_obs.bin')

# scale_og = torch.cuda.FloatTensor(np.fromfile('./genetic_best_scale_multicross_1000_1_4.bin',dtype=np.float32))


class Optimizer():
    """Class that implements genetic algorithm."""

    def __init__(self, length, retain=0.3,
                 random_select=0.1, mutate_chance=0.2):
        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.len = length

    def create_population(self, count):
        """Create a population of random scale.
        Args:
            count (int): Number of scale to generate, aka the
                size of the population
        Returns:
            (list): Population of scale objects
        """
        n = tdist.uniform.Uniform(torch.Tensor([0.9]),torch.Tensor([1.0]))

        pop = []
        for _ in range(0, count):
            # Create a random scale.
            scale = n.sample((self.len,)).reshape(1,-1).squeeze().cuda()
            #scale = torch.clamp(scale,0.98,1.02)
            # Add the scale to our population.
            pop.append(scale)

        return pop

    @staticmethod
    def fitness(scale):
        scale = scale.cuda()
        x_var = torch.mul(scale,torch.cuda.FloatTensor(target_obs[:,0]))
        y_var = torch.mul(scale,torch.cuda.FloatTensor(target_obs[:,1]))
        z_var = torch.mul(scale,torch.cuda.FloatTensor(target_obs[:,2]))
        i_var = torch.cuda.FloatTensor(target_obs[:,3])

        x_final = torch.cuda.FloatTensor(PCL_except_car[:,0])
        y_final = torch.cuda.FloatTensor(PCL_except_car[:,1])
        z_final = torch.cuda.FloatTensor(PCL_except_car[:,2])
        i_final = torch.cuda.FloatTensor(PCL_except_car[:,3])

        x_final = torch.cat([x_final,x_var],dim = 0)
        y_final = torch.cat([y_final,y_var],dim = 0)
        z_final = torch.cat([z_final,z_var],dim = 0)
        i_final = torch.cat([i_final,i_var],dim = 0)

        grids = xyzi2grid(x_final, y_final, z_final, i_final)
        FM = gridi2feature(grids)


        with torch.no_grad():
            outputPytorch = pytorchModels(FM)
        lossValue,loss_object,loss_distance = loss.lossPassiveAttack(outputPytorch,x_var,y_var,z_var,scale)

        del x_var,y_var,z_var,i_var,x_final,y_final,z_final,i_final,outputPytorch
        torch.cuda.empty_cache()

        return -lossValue


    def grade(self, pop):
        """Find average fitness for a population.
        Args:
            pop (list): The population of scale
        Returns:
            (float): The average accuracy of the population
        """
        fitlist = []
        summed = []
        for scale in pop:
            fitlist.append((self.fitness(scale),scale)) 
            summed.append(fitlist[-1][0])

        return sum(summed) / float((len(pop))), fitlist

    def breed(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother : scale objects
            father : scale objects
        Returns:
            (list): Two scale objects
        """
        children = []
        for _ in range(2):

            child = torch.zeros(self.len)

            # Loop through the DNA and pick for the kid.
            for idx in range(self.len):
                child[idx] = random.choice(
                    [mother[idx], father[idx]]
                )
            # cross = random.randint(0,self.len-1)
            # child[:cross] = mother[:cross]
            # child[cross:] = father[cross:]

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                child = self.mutate(child)

            children.append(child)

        return children

    def mutate(self, scale):
        """Randomly mutate one part of the network.
        Args:
            network (dict): The network parameters to mutate
        Returns:
            (Network): A randomly mutated network object
        """
        # Choose a random key.
        mutation = random.randint(0,self.len-1)

        # Mutate one of the params.
        scale[mutation] = 2. - scale[mutation]

        return scale

    def evolve(self, pop, fitlist):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """
        # Get scores for each network.
        graded = fitlist

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents


if __name__ == '__main__':

    generations = 25
    population = 2000

    for retain in [0.1,0.2,0.3]:
        for rand in [0.01,0.05,0.1]:
            optimizer = Optimizer(target_obs.shape[0],retain=retain,random_select=rand)
            scales = optimizer.create_population(population)

            print "***Doing retain %f, random %f***" % (retain, rand)

            for i in range(generations):

                print "***Doing generation %d of %d***" % (i + 1, generations)


                # Get the average accuracy for this generation.
                average_fitness, fitlist = optimizer.grade(scales)
                rank = [x for x in sorted(fitlist, key=lambda x: x[0], reverse=True)]

                # Print out the average accuracy each generation.
                print "Generation average: %.5f, best fitness:  %.5f" % (average_fitness,rank[0][0])
                # Evolve, except on the last iteration.
                if i != generations - 1:
                    # Do the evolution.
                    scales = optimizer.evolve(scales,fitlist)


                # os.mkdir('./genetic/%d'%(dataIdx))

                rank[0][1].cpu().numpy().tofile('./genetic/%f_%f_2000_101_37_car_09_10.bin'%(retain,rand))
