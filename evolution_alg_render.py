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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

# scale_og = torch.cuda.FloatTensor(np.fromfile('./genetic_best_scale_multicross_1000_1_4.bin',dtype=np.float32))

with torch.no_grad():
    meshes_og = torch.nn.functional.embedding(face,vertex_og)
    edge_1_og = meshes_og[:,1] - meshes_og[:,0]
    edge_2_og = meshes_og[:,2] - meshes_og[:,0]
    edge_3_og = meshes_og[:,1] - meshes_og[:,2]

    dis_og = torch.stack([torch.sqrt(torch.pow(edge_1_og[:,0],2) + 
                torch.pow(edge_1_og[:,1],2) +
                torch.pow(edge_1_og[:,2],2)), torch.sqrt(torch.pow(edge_2_og[:,0],2) + 
                torch.pow(edge_2_og[:,1],2) +
                torch.pow(edge_2_og[:,2],2)), torch.sqrt(torch.pow(edge_3_og[:,0],2) + 
                torch.pow(edge_3_og[:,1],2) +
                torch.pow(edge_3_og[:,2],2))],dim = 1)

class Optimizer():
    """Class that implements genetic algorithm."""

    def __init__(self, shape, retain=0.01,
                 random_select=0.002, mutate_chance=0.2):
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

    def create_population(self, count, n):
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
        lossValue,loss_object,loss_distance = loss.lossRenderAttack(outputPytorch,vertex,vertex_og,face,0.001,dis_og) 
        torch.cuda.empty_cache()

        return -loss_object


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

    def breed(self, parent, generation,n):


        child = torch.zeros(self.shape).cuda()
        # print n1.sample((self.len,)).reshape(1,-1).squeeze().cuda()
        # Loop through the DNA and pick for the kid.
        child = parent + n.sample(self.shape).reshape(-1,3).squeeze().cuda()
        # print child
        # child = torch.clamp(child,0.9,1.1)
        # cross = random.randint(0,self.len-1)
        # child[:cross] = mother[:cross]
        # child[cross:] = father[cross:]

        # Randomly mutate some of the children.
        # if self.mutate_chance > random.random():
        #     child = self.mutate(child)

        return child


    def evolve(self, pop, fitlist, generation, n):
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

            # Breed them.
            parent = parents[parent]
            baby = self.breed(parent,generation,n)

            if len(children) < desired_length:
                children.append(baby)

        parents.extend(children)

        return parents


if __name__ == '__main__':

    generations = 50
    population = 500

    for perturbation in [0.1,0.15,0.2]:

        print "***Doing perturbation %f ***" % (perturbation)
        n = tdist.Normal(torch.tensor([0.0]), torch.tensor([perturbation]))
        optimizer = Optimizer(vertex1.shape)
        vertices = optimizer.create_population(population,n)

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
                vertices = optimizer.evolve(vertices,fitlist,i,n)


            # os.mkdir('./genetic/%d'%(dataIdx))

            rank[0][1].cpu().numpy().tofile('./evolution/%f_500_1_render_05.bin'%(perturbation))