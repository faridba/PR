#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
import numpy as np
from probabilistic_lib.functions import angle_wrap, comp, state_inv, state_inv_jacobian, compInv
import scipy.linalg
import rospy

#============================================================================
class EKF_SLAM(object):
    '''
    Class to hold the whole EKF-SLAM.
    '''
    
    #========================================================================
    def __init__(self, x0,y0,theta0, odom_lin_sigma, 
                 odom_ang_sigma, meas_rng_noise, meas_ang_noise):
        '''
        Initializes the ekf filter
        room_map : an array of lines in the form [x1 y1 x2 y2]
        num      : number of particles to use
        odom_lin_sigma: odometry linear noise
        odom_ang_sigma: odometry angular noise
        meas_rng_noise: measurement linear noise
        meas_ang_noise: measurement angular noise
        '''
        
        # Copy parameters
        self.odom_lin_sigma = odom_lin_sigma
        self.odom_ang_sigma = odom_ang_sigma
        self.meas_rng_noise = meas_rng_noise
        self.meas_ang_noise = meas_ang_noise
        self.chi_thres = 0.4 # TODO chose your own value
       
        # Odometry uncertainty 
        self.Qk = np.array([[ self.odom_lin_sigma**2, 0, 0],\
                            [ 0, self.odom_lin_sigma**2, 0 ],\
                            [ 0, 0, self.odom_ang_sigma**2]])
        
        # Measurements uncertainty
        self.Rk=np.eye(2)
        self.Rk[0,0] = self.meas_rng_noise
        self.Rk[1,1] = self.meas_ang_noise
        
        # State vector initialization
        self.xk = np.array([x0,y0,theta0]) # Position
        self.Pk = np.zeros((3,3)) # Uncertainty
        
        # Initialize buffer for forcing observing n times a feature before 
        # adding it to the map
        self.featureObservedN = np.array([])
        self.min_observations = 5
    
    #========================================================================
    def get_number_of_features_in_map(self):
        '''
        returns the number of features in the map
        '''
        return (self.xk.size-3)/2
    
    #========================================================================
    def get_polar_line(self, line, odom):
        '''
        Transforms a line from [x1 y1 x2 y2] from the world frame to the
        vehicle frame using odometry [x y ang].
        Returns [range theta]
        '''
        # Line points
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        
        # Compute line (a, b, c) and range
        line = np.array([y1-y2, x2-x1, x1*y2-x2*y1])
        pt = np.array([odom[0], odom[1], 1])
        dist = np.dot(pt, line) / np.linalg.norm(line[:2])
        
        # Compute angle
        if dist < 0:
            ang = np.arctan2(line[1], line[0])
        else:
            ang = np.arctan2(-line[1], -line[0])
        
        # Return in the vehicle frame
        return np.array([np.abs(dist), angle_wrap(ang - odom[2])])
        
    #========================================================================
    def predict(self, uk):
        
        '''
        Predicts the position of the robot according to the previous position and the odometry measurements. It also updates the uncertainty of the position
        '''
        #TODO: Program this function
        # - Update self.xk and self.Pk using uk and self.Qk

        # Compound robot with odometry
        a=comp(self.xk[0:3],uk)
        self.xk=np.append(a, self.xk[3:])
        # Compute jacobians of the composition with respect to robot (A_k) 
        # and odometry (W_k)
	W=np.array([[np.cos(self.xk[2]), -np.sin(self.xk[2]), 0],[np.sin(self.xk[2]), np.cos(self.xk[2]), 0],[0, 0, 1]])
	A=np.array([[1, 0, -np.dot(np.sin(self.xk[2]),uk[0]) - np.dot(np.cos(self.xk[2]),uk[1])],[0, 1, np.dot(np.cos(self.xk[2]),uk[0]) - np.dot(np.sin(self.xk[2]),uk[1])],[0, 0, 1]])
        
	n=self.get_number_of_features_in_map()
        # Prepare the F_k and G_k matrix for the new uncertainty computation
	F_k=np.eye(3+2*n)
	G_k=np.zeros((3+2*n,3))
	F_k[0:3,0:3]=A		
	G_k[0:3,0:3]=W		
        # Compute uncertainty
        # Update the class variables
        self.Pk=np.dot(np.dot(F_k,self.Pk),F_k.T)+np.dot(np.dot(G_k,self.Qk),G_k.T)
    #========================================================================
        
    def data_association(self, lines):
        '''
        Implements ICCN for each feature of the scan.
        '''
    
        #TODO: Program this function
        # fore each sensed line do:
        #   1- Transform the sened line to polar
        #   2- for each feature of the map (in the state vector) compute the 
        #      mahalanobis distance
        #   3- Data association
        # Init variable
        Innovk_List   = list()
        H_k_List      = list()
        Rk_List       = list()
        idx_not_associated = np.array([])

	n=self.get_number_of_features_in_map()

	for i in range(lines.shape[0]):
	    z = self.get_polar_line(lines[i,:], np.array([0, 0, 0]) )

            # Variables for finding minimum
            minD = 1e9
            minj = -1

            for j in range(n):
                (D,v,h,H)=self.lineDist(z,j)
		
                # Check if the obseved line is the one with smallest
                # mahalanobis distance
                if np.sqrt(D) < minD:
                    minj = j
                    minz = z
                    minh = h
                    minH = H
                    minv = v
                    minD = D

            # Minimum distance below threshold
            if minD < self.chi_thres:
                # Append results
		self.featureObservedN[minj]=self.featureObservedN[minj]+1
                H_k_List.append(minH)
                Innovk_List.append(minv)
                Rk_List.append(self.Rk)
	    else:
                idx_not_associated=np.append(idx_not_associated,i)
		
                
        return Innovk_List, H_k_List, Rk_List, idx_not_associated
        
    #========================================================================
    def update_position(self, Innovk_List, H_k_List, Rk_List) :
        '''
        Updates the position of the robot according to the given the position
        and the data association parameters.
        Returns state vector and uncertainty.
        
        '''
        #TODO: Program this function
        if len(Innovk_List)<1:
            return
        n=(self.xk.shape[0]-3)/2
        m = len(H_k_List)
        H = np.zeros((2*m, 2*n+3))
        v = np.zeros((2*m))
        R = np.zeros((2*m, 2*m))
        for i in range(m):
            R[2*i:2*i+2, 2*i:2*i+2] = Rk_List[i]
            v[2*i:2*i+2] = Innovk_List[i]
            H[2*i:2*i+2, :] = H_k_List[i]
	
        # There is data to update
        if not m > 0:
            return

        # TODO: Program this function
        ################################################################
        # Do the EKF update
        S=np.dot(np.dot(H,self.Pk),H.T)+R
	I=np.eye(self.xk.shape[0])
        k =np.dot(np.dot(self.Pk,H.T),np.linalg.inv(S))
	A=np.dot(k,v)
	#direct addition was not working for some reason
	for i in range(self.xk.shape[0]):
	    self.xk[i]=self.xk[i]+A[0,i]
        
	self.Pk=np.dot(np.dot((I-np.dot(k,H)),self.Pk),(I-np.dot(k,H)).T) + np.dot(np.dot(k,R),k.T)
        # Kalman Gain
        
        # Update Position
        
        # Update Uncertainty
            
    #========================================================================
    def state_augmentation(self, lines, idx):
        '''
        given the whole set of lines read by the kineckt sensor and the
        indexes that have not been associated augment the state vector to 
        introduce the new features
        '''
        # If no features to add to the map exit function
        if idx.size<1:
            return
        # TODO Program this function
	n=self.get_number_of_features_in_map()
        F_k=np.eye(3+2*n)
        G_k=np.zeros((3+2*n,2))
	print'a'
	print idx
	print lines
        for i in range(idx.size):
	    A=idx[i]
            z = self.get_polar_line(lines[idx[i],:], np.array([0, 0, 0]) )
            (obsW, H_tf, H_line)=self.tfPolarLine(self.xk[0:3],z)

	    self.featureObservedN = np.append(self.featureObservedN,1)
	     	    

            self.xk=np.append(self.xk, obsW)

	    F_k=np.vstack((F_k, np.hstack((np.zeros((2,2*n)),H_tf)) ))
	    G_k=np.vstack((G_k,H_line))

        self.Pk=np.dot(np.dot(F_k,self.Pk),F_k.T)+np.dot(np.dot(G_k,self.Rk),G_k.T)

    #========================================================================
    def tfPolarLine(self,tf,line):
        '''
        Transforms a polar line in the robot frame to a polar line in the
        world frame
        '''
        # Decompose transformation
        x_x = tf[0]
        x_y = tf[1]
        x_ang = tf[2]  
        
        # Compute the new phi
        phi = angle_wrap(line[1] + x_ang)
        
        # Auxiliar computations
        sqrt2 = x_x**2+x_y**2
        sqrt = np.sqrt(sqrt2)
        atan2 = np.arctan2(x_y,x_x)
        sin = np.sin(atan2 - phi)
        cos = np.cos(atan2 - phi)
        
        # Compute the new rho
        rho = line[0] + sqrt* cos  
        sign = 1
        if rho <0:
            rho = -rho
            phi = angle_wrap(phi+pi)         
            sign = -1
        
        # Allocate jacobians
        H_tf = np.zeros((2,3))
        H_line = np.eye(2)

        a=np.cos(angle_wrap(line[1]+x_ang))
        b=np.sin(angle_wrap(line[1]+x_ang))
        c=-x_x*np.sin(angle_wrap(line[1]+x_ang))+x_y*np.cos(angle_wrap(line[1]+x_ang))

        # TODO: Evaluate jacobian respect to transformation
        H_tf=np.mat([[sign*a, sign*b, sign*c],[0, 0, 1]])

        # TODO: Evaluate jacobian respect to line
        H_line=np.mat([[sign*1, sign*c],[0, 1]])
                
        return np.array([rho,phi]), H_tf, H_line
                
    #========================================================================
    def lineDist(self,z,idx):
        '''
        Given a line and an index of the state vector it computes the
        distance between both lines
        '''        
        # TODO program this function
                
        # Transform the map line into robot frame and compute jacobians
	#
	n=self.get_number_of_features_in_map()
        (x,J)=compInv(self.xk[0:3])
        (h, H_position, H_line)=self.tfPolarLine(x,self.xk[3+2*idx:3+2*idx+2])
        
        # Allocate overall jacobian
	H=np.zeros((2,3+2*n))
        A=np.dot(H_position,J)
                
        # Concatenate position jacobians and place them into the position
	H[0:2,0:3]=A
        
        # Place the position of the jacobina with respec to the line in its
        # position
	H[0:2,3+2*idx:3+2*idx+2]=H_line
        
        # Calculate innovation
        v = z-h
        
        # Calculate innovation uncertainty
        S = self.Rk+np.dot(np.dot(H,self.Pk),H.T)
  
        # Calculate mahalanobis distance
        D =(np.dot(np.dot(v.T , np.linalg.inv(S)) , v))
        
        return D,v,h,H
