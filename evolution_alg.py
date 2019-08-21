from functools import reduce
from operator import add
import random
import c2p_segmentation
import torch
import torch.distributions as tdist
import loss
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

root_path = './cnnseg/velodyne64/'
pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
protofile = root_path + 'deploy.prototxt'
weightfile = root_path + 'deploy.caffemodel'
pytorchModels = c2p_segmentation.generatePytorch(protofile, weightfile)
dataIdx = 7
pclfile = '%06d.bin'%(dataIdx)
PCL_path = pclpath + pclfile
_, PCL_except_car,target_obs = c2p_segmentation.preProcess(PCL_path,'./7/18_obs.bin')

# scale_og = torch.cuda.FloatTensor(np.fromfile('./genetic_best_scale_multicross_1000_1_4.bin',dtype=np.float32))


class Optimizer():
    """Class that implements genetic algorithm."""

    def __init__(self, length, retain=0.01,
                 random_select=0.1, mutate_chance=0.1):
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
        #n = tdist.Normal(torch.tensor([1.0]), torch.tensor([0.03]))
        n = tdist.uniform.Uniform(torch.Tensor([0.9]),torch.Tensor([1.1]))

        pop = []
        for _ in range(0, count):
            # Create a random scale.
            scale = n.sample((self.len,)).reshape(1,-1).squeeze().cuda()
            # scale = torch.clamp(scale,0.95,1.05)
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

        PCL = torch.stack([x_final,y_final,z_final,i_final]).permute(1,0).cpu().detach().numpy()
        PCLConverted = c2p_segmentation.mapPointToGrid(PCL)
        featureM = c2p_segmentation.generateFM(PCL, PCLConverted)
        featureM = np.array(featureM).astype('float32')
        featureM = torch.cuda.FloatTensor(featureM)
        featureM = featureM.view(1,6,672,672)
        with torch.no_grad():
            outputPytorch = pytorchModels(featureM)
        lossValue,loss_object,loss_distance = loss.lossPassiveAttack(outputPytorch,x_var,y_var,z_var,scale)

        del x_var,y_var,z_var,i_var,x_final,y_final,z_final,i_final,PCL,PCLConverted,featureM,outputPytorch
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

    def breed(self, parent, generation):


        child = torch.zeros(self.len)

        n1 = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.05-generation/1500]))
        # print n1.sample((self.len,)).reshape(1,-1).squeeze().cuda()
        # Loop through the DNA and pick for the kid.
        child = parent + n1.sample((self.len,)).reshape(1,-1).squeeze().cuda()
        # print child
        child = torch.clamp(child,0.9,1.1)
        # cross = random.randint(0,self.len-1)
        # child[:cross] = mother[:cross]
        # child[cross:] = father[cross:]

        # Randomly mutate some of the children.
        # if self.mutate_chance > random.random():
        #     child = self.mutate(child)

        return child

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

    def evolve(self, pop, fitlist, generation):
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

        # Add children
        while len(children) < desired_length:

            # Get a random mom and dad.
            parent = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...

            # Breed them.
            parent = parents[parent]
            baby = self.breed(parent,generation)

            if len(children) < desired_length:
                children.append(baby)

        parents.extend(children)

        return parents


if __name__ == '__main__':

    generations = 50
    population = 1000
    optimizer = Optimizer(target_obs.shape[0])
    scales = optimizer.create_population(population)

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
            scales = optimizer.evolve(scales,fitlist,i)

        rank[0][1].cpu().numpy().tofile('evolution_best_scale_multicross_1000_7_18_cyl_09_11.bin')
