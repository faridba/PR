#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from probabilistic_lib.functions import angle_wrap, get_polar_line

#===============================================================================
class ParticleFilter(object):
    '''
    Class to hold the whole particle filter.
    
    p_wei: weights of particles in array of shape (N,)
    p_ang: angle in radians of each particle with respect of world axis, shape (N,)
    p_xy : position in the world frame of the particles, shape (2,N)
    '''
    
    #===========================================================================
    def __init__(self, room_map, num, odom_lin_sigma, odom_ang_sigma, 
                 meas_rng_noise, meas_ang_noise,x_init,y_init,theta_init):
        '''
        Initializes the particle filter
        room_map : an array of lines in the form [x1 y1 x2 y2]
        num      : number of particles to use
        odom_lin_sigma: odometry linear noise
        odom_ang_sigma: odometry angular noise
        meas_rng_noise: measurement linear noise
        meas_ang_noise: measurement angular noise
        '''
        
        # Copy parameters
        self.map = room_map
        self.num = num
        self.odom_lin_sigma = odom_lin_sigma
        self.odom_ang_sigma = odom_ang_sigma
        self.meas_rng_noise = meas_rng_noise
        self.meas_ang_noise = meas_ang_noise
        
        # Map
        map_xmin = np.min(self.map[:, 0])
        map_xmax = np.max(self.map[:, 0])
        map_ymin = np.min(self.map[:, 1])
        map_ymax = np.max(self.map[:, 1])
        
        # Particle initialization arround starting point
        self.p_wei = 1.0 / num * np.ones(num)
        self.p_ang =2 * np.pi * np.random.rand(num)
        self.p_xy  = np.vstack(( x_init+ 1*np.random.rand(num)-0.5,
                                 y_init+ 1*np.random.rand(num)-0.5 ))
        #Flags for resampling                         
        self.moving=False
        self.n_eff=0 #Initialize Efficent number as 0
    
    #===========================================================================
    def predict(self, odom):
        '''
        Moves particles with the given odometry.
        odom: incremental odometry [delta_x delta_y delta_yaw] in the vehicle frame
        '''
        #Check if we have moved from previous reading.
        if odom[0]==0 and odom[1]==0 and odom[2]==0:
            self.moving=False
        else:
            # TODO: code here;!!
            # Add Gaussian noise to odometry measures
            odomx=odom[0] * np.cos(self.p_ang) - odom[1] * np.sin(self.p_ang)
            odomy=odom[0] * np.sin(self.p_ang) + odom[1] * np.cos(self.p_ang)

            odomx+=self.odom_lin_sigma*np.random.randn(self.p_xy.shape[1])
            odomy+=self.odom_lin_sigma*np.random.randn(self.p_xy.shape[1])
            odomt=odom[2]+self.odom_ang_sigma*np.random.randn(self.p_ang.shape[0])          

            # Increment particle positions in correct frame
            self.p_xy[0]+=odomx
            self.p_xy[1]+=odomy

            # Increment angle
            self.p_ang+=odomt
            self.p_ang = angle_wrap(self.p_ang)
            
            #Update flag for resampling
            self.moving=True     
    
    #===========================================================================
    def weight(self, lines):
        '''
        Look for the lines seen from the robot and compare them to the given map.
        Lines expressed as [x1 y1 x2 y2].
        '''
        # TODO: code here!!
        # Constant values for all weightings
        val_rng = 1.0 / (self.meas_rng_noise * np.sqrt(2 * np.pi))
        val_ang = 1.0 / (self.meas_ang_noise * np.sqrt(2 * np.pi))
        
        # Loop over particles
        for i in range(self.num):
            odompass=[self.p_xy[0,i], self.p_xy[1,i], self.p_ang[i]]
            # Transform map lines to local frame and to [range theta]
            room=np.zeros((self.map.shape[0],2))

            for j in range(self.map.shape[0]):
                room[j,:]= get_polar_line(self.map[j,:],odompass)
                
            # Transform measured lines to [range theta] and weight them
            #print lines
            for j in range(lines.shape[0]):
                meas= get_polar_line(lines[j,:])
                # Weight them
                wr=val_rng*np.exp(-np.power((meas[0]-room[:,0]),2)/(2*self.meas_rng_noise*self.meas_rng_noise))
                wt=val_ang*np.exp(-np.power((meas[1]-room[:,1]),2)/(2*self.meas_rng_noise*self.meas_rng_noise))
                #optional part
                # make sure segments correspond, if not put weight to zero
                for k in range(room.shape[0]):
	            rooml=math.sqrt((self.map[k,1]-self.map[k,3])*(self.map[k,1]-self.map[k,3]) + (self.map[k,2]-self.map[k,0])*(self.map[k,2]-self.map[k,0]))
                    measl=math.sqrt((lines[j,1]-lines[j,3])*(lines[j,1]-lines[j,3]) + (lines[j,2]-lines[j,0])*(lines[j,2]-lines[j,0]))
		    if rooml<measl:
                        wr[k]=0
                        wt[k]=0
                # Take best weighting (best associated lines)
                self.p_wei[i]*=np.max(wr*wt)
            
        # Normalize weights
        self.p_wei /= np.sum(self.p_wei)
        # TODO: Compute efficient number
        self.n_eff=1/np.sum(self.p_wei*self.p_wei)
        
    #===========================================================================
    def resample(self):
        '''
        Systematic resampling of the particles.
        '''
        # TODO: code here!!
        # Look for particles to replicate
        xy=self.p_xy
        ang=self.p_ang
        r=np.random.uniform(0.0,1.0)/(1.0*self.num)
        c=self.p_wei[1]
        m=0

        for i in range(self.num):
            u=r+((1.0*i)/(self.num*1.0) )
            while u>c:
                m=m+1
		m=m%self.num
                c+=self.p_wei[m]
            ang[i]=self.p_ang[m]
            xy[:,i]=self.p_xy[:,m]
        
        # Pick chosen particles
        self.p_xy =xy
        self.p_ang =ang
        self.p_wei =1.0 / self.num * np.ones(self.num)
    
    #===========================================================================
    def get_mean_particle(self):
        '''
        Gets mean particle.
        '''
        # Weighted mean
        weig = np.vstack((self.p_wei, self.p_wei))
        mean = np.sum(self.p_xy * weig, axis=1) / np.sum(self.p_wei)
        
        ang = np.arctan2( np.sum(self.p_wei * np.sin(self.p_ang)) / np.sum(self.p_wei),
                          np.sum(self.p_wei * np.cos(self.p_ang)) / np.sum(self.p_wei) )
                          
        return np.array([mean[0], mean[1], ang])
