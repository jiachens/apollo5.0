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
import render_new

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

root_path = './cnnseg/velodyne64/'
pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
protofile = root_path + 'deploy.prototxt'
weightfile = root_path + 'deploy.caffemodel'
pytorchModel = c2p_segmentation.generatePytorch(protofile, weightfile)
dataIdx = 1
pclfile = '%06d.bin'%(dataIdx)
PCL_path = pclpath + pclfile
PCL = c2p_segmentation.loadPCL(PCL_path,True)                

x_final = torch.cuda.FloatTensor(PCL[:,0])
y_final = torch.cuda.FloatTensor(PCL[:,1])
z_final = torch.cuda.FloatTensor(PCL[:,2])
i_final = torch.cuda.FloatTensor(PCL[:,3])
ray_direction, length = render_new.get_ray(x_final,y_final,z_final)
vertex1,face = render_new.loadmesh('./cube248.ply')
vertex_og = vertex1.clone()
n = tdist.uniform.Uniform(torch.Tensor([-0.2]),torch.Tensor([0.2]))
# scale_og = torch.cuda.FloatTensor(np.fromfile('./genetic_best_scale_multicross_1000_1_4.bin',dtype=np.float32))


class Optimizer():
    """Class that implements genetic algorithm."""

    def __init__(self, shape, retain=0.3,
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
        self.shape = shape

    def create_population(self, count):
        """Create a population of random scale.
        Args:
            count (int): Number of scale to generate, aka the
                size of the population
        Returns:
            (list): Population of scale objects
        """
        pop = []
        for _ in range(0, count):
            # Create a random scale.
            vertex = n.sample(self.shape).reshape(-1,3).squeeze().cuda() + vertex1
            #scale = torch.clamp(scale,0.98,1.02)
            # Add the scale to our population.
            pop.append(vertex)

        return pop

    @staticmethod
    def fitness(vertex):

        point_cloud = render_new.render(ray_direction,length,vertex,face,i_final)
        grid = xyzi2grid(point_cloud[:,0],point_cloud[:,1],point_cloud[:,2],point_cloud[:,3])
        featureM = gridi2feature(grid)    
        with torch.no_grad():
            outputPytorch = pytorchModel(featureM)
        lossValue,loss_object,loss_distance = loss.lossRenderAttack(outputPytorch,vertex,vertex_og,0.05) 
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
        for vertex in pop:
            fitlist.append((self.fitness(vertex),vertex)) 
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

            child = torch.zeros(self.shape).cuda()

            # Loop through the DNA and pick for the kid.
            for idx in range(self.shape[0]):
                child[idx,:] = random.choice(
                    [mother[idx,:], father[idx,:]]
                )
            # cross = random.randint(0,self.len-1)
            # child[:cross] = mother[:cross]
            # child[cross:] = father[cross:]

            # Randomly mutate some of the children.
            # if self.mutate_chance > random.random():
            #     child = self.mutate(child)

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
            optimizer = Optimizer(vertex1.shape,retain=retain,random_select=rand)
            vertices = optimizer.create_population(population)

            print "***Doing retain %f, random %f***" % (retain, rand)

            for i in range(generations):

                print "***Doing generation %d of %d***" % (i + 1, generations)


                # Get the average accuracy for this generation.
                average_fitness, fitlist = optimizer.grade(vertices)
                rank = [x for x in sorted(fitlist, key=lambda x: x[0], reverse=True)]

                # Print out the average accuracy each generation.
                print "Generation average: %.5f, best fitness:  %.5f" % (average_fitness,rank[0][0])
                # Evolve, except on the last iteration.
                if i != generations - 1:
                    # Do the evolution.
                    vertices = optimizer.evolve(vertices,fitlist)


                # os.mkdir('./genetic/%d'%(dataIdx))

                rank[0][1].cpu().numpy().tofile('./genetic/%f_%f_2000_1_render_05.bin'%(retain,rand))
