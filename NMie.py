# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:04:20 2017

@author: Bernhard Steinhauser

Mie[1] code to calculate the cross sections for a core-shell structure with an arbitrary number of shells.
The iteration scheme used is based on an approach by Quinten et.al.[2]

Needs scipy and numpy installed

[1] Gustav Mie, Beiträge zur Optik trüber Medien, speziell kolloidaler Metallösungen, Ann. Phys., 1908, 330(3)
[2] J. Sinzig, M. Quinten, Scattering and absorption by spherical multilayer particles, Appl. Phys. A, 1994, 58(2)
"""

import numpy as np
import math as m
from scipy.special import jv as bessel
from scipy.special import yv as neumann
from scipy.special import hankel1 as hankel


class NMie: # Defines a new class called NMie that is the main class for all EM calculations
    def __init__( self , **kwargs ): # initilization of NMie with the diameters and the refractive indices of the core and the shell
        self.diameters = kwargs.get( 'diameters' )
        self.refind    = np.transpose( kwargs.get( 'refind' ) )  # refractive indices for the shells. It must have r+1 entries for all the shells + host medium
        self.wl        = kwargs.get( 'wl' ) 
        self.numshells = np.size(self.diameters)

        # pre allocation of variables
        self.x         = np.zeros( ( self.wl.size , self.numshells ) ) * 1j
        self.m         = np.zeros( ( self.wl.size , self.numshells ) ) * 1j 

        # Calculation of reduced parameters x and m
        self.__redparameters()
        # Calculation of number of modes to be calculated for the given parameter set
        self.nummodes = self.__calcNumModes( )

        # pre allocation of variables
        self.Sn        = np.zeros( ( self.wl.size ) ) * 1j
        self.Tn        = np.zeros( ( self.wl.size ) ) * 1j
        self.an        = np.zeros( ( self.wl.size ) ) * 1j
        self.bn        = np.zeros( ( self.wl.size ) ) * 1j

        self.qsca_el   = np.zeros( ( self.wl.size , self.nummodes ) )
        self.qsca_ma   = np.zeros( ( self.wl.size , self.nummodes ) )
        self.qabs_el   = np.zeros( ( self.wl.size , self.nummodes ) )
        self.qabs_ma   = np.zeros( ( self.wl.size , self.nummodes ) )        
        self.qext_el   = np.zeros( ( self.wl.size , self.nummodes ) )
        self.qext_ma   = np.zeros( ( self.wl.size , self.nummodes ) )
        
        self.qsca      = np.zeros( ( self.wl.size ) )
        self.qext      = np.zeros( ( self.wl.size ) )
        self.qabs      = np.zeros( ( self.wl.size ) )
        
        
        
    def __calcNumModes( self ):
    # Calculation of number of modes to be calculated for the given parameter set
        maxx = np.abs( np.amax( self.x ) )
        dmy = 0
        if maxx < 8:
            dmy = maxx + 4 * maxx**( 1.0/3.0 ) + 1
        elif maxx < 4200:
            dmy = maxx + 4.05 * maxx**( 1.0/3.0 ) + 2
        else:
            dmy = maxx + 4 * maxx**( 1.0/3.0 ) + 2
        return int( m.ceil( dmy ) )
        
    def __redparameters( self ):
    # Calculation of reduced parameters x and m
        dmy = 0
        for s in range( self.numshells ):
            dmy += self.diameters[ s ]
            self.x[ : , s ] = 2 * np.pi / self.wl *  self.refind[ : , s + 1 ] * dmy
            self.m[ : , s ] = self.refind[ : , s ] / self.refind[ : , s + 1 ]

    def __an( self , n ):
        
        xr = self.x[ : , - 1 ]
        mr = self.m[ : , - 1 ]
        r = self.numshells 

        nom    =       self.__Psi( xr , n ) * ( self.__DPsi( mr * xr , n ) + self.__Sn( r - 1 , n ) * self.__DChi( mr * xr , n ) )
        nom   -= mr * self.__DPsi( xr , n ) * (  self.__Psi( mr * xr , n ) + self.__Sn( r - 1 , n ) *  self.__Chi( mr * xr , n ) )
        denom  =        self.__Xi( xr , n ) * ( self.__DPsi( mr * xr , n ) + self.__Sn( r - 1 , n ) * self.__DChi( mr * xr , n ) )
        denom -= mr *  self.__DXi( xr , n ) * (  self.__Psi( mr * xr , n ) + self.__Sn( r - 1 , n ) *  self.__Chi( mr * xr , n ) )

        self.an = nom / denom

        return nom / denom
        
    def __bn( self , n ):
                    
                   
        xr = self.x[ : , - 1 ]
        mr = self.m[ : , - 1 ]
        r = self.numshells 

        nom    = mr * self.__Psi( xr , n ) * ( self.__DPsi( mr * xr , n ) + self.__Tn( r - 1 , n ) * self.__DChi( mr * xr , n ) ) 
        nom   -=     self.__DPsi( xr , n ) * (  self.__Psi( mr * xr , n ) + self.__Tn( r - 1 , n ) *  self.__Chi( mr * xr , n ) )
        denom  = mr *  self.__Xi( xr , n ) * ( self.__DPsi( mr * xr , n ) + self.__Tn( r - 1 , n ) * self.__DChi( mr * xr , n ) ) 
        denom -=      self.__DXi( xr , n ) * (  self.__Psi( mr * xr , n ) + self.__Tn( r - 1 , n ) *  self.__Chi( mr * xr , n ) )
        
        self.bn = nom / denom
        return nom / denom
        
    def __Sn( self , S , n ):
        Sn = np.zeros( ( self.wl.size ) )        
        
        if( S > 0 ):
            xs = self.x[ : , S - 1 ]
            ms = self.m[ : , S - 1 ]
            
            nom    =       self.__Psi( xs , n ) * ( self.__DPsi( ms * xs , n ) + self.__Sn( S - 1 , n ) * self.__DChi( ms * xs , n ) )
            nom   -= ms * self.__DPsi( xs , n ) * (  self.__Psi( ms * xs , n ) + self.__Sn( S - 1 , n ) *  self.__Chi( ms * xs , n ) ) 
            denom  =       self.__Chi( xs , n ) * ( self.__DPsi( ms * xs , n ) + self.__Sn( S - 1 , n ) * self.__DChi( ms * xs , n ) )
            denom -= ms * self.__DChi( xs , n ) * (  self.__Psi( ms * xs , n ) + self.__Sn( S - 1 , n ) *  self.__Chi( ms * xs , n ) )
            Sn = - nom/denom

        self.Sn = Sn
        return Sn

    def __Tn( self , S , n ):
        
        Tn = np.zeros( ( self.wl.size ) ) 
            
        if( S > 0 ):
            xs = self.x[ : , S - 1 ]
            ms = self.m[ : , S - 1 ]
            
            nom    = ms * self.__Psi( xs , n ) * ( self.__DPsi( ms * xs , n ) + self.__Tn( S - 1 , n ) * self.__DChi( ms * xs , n ) ) 
            nom   -=     self.__DPsi( xs , n ) * (  self.__Psi( ms * xs , n ) + self.__Tn( S - 1 , n ) *  self.__Chi( ms * xs , n ) ) 
            denom  = ms * self.__Chi( xs , n ) * ( self.__DPsi( ms * xs , n ) + self.__Tn( S - 1 , n ) * self.__DChi( ms * xs , n ) ) 
            denom -=     self.__DChi( xs , n ) * (  self.__Psi( ms * xs , n ) + self.__Tn( S - 1 , n ) *  self.__Chi( ms * xs , n ) )
            Tn = - nom/denom

        self.Tn = Tn   
        return Tn
    
    def calculate_crosssections( self ):
        
        for n in range( 1 , self.nummodes , 1 ):
            
            an = self.__an( n )
            bn = self.__bn( n )
            k = 2 * np.pi * np.real(self.refind[ : , -1 ]) / self.wl
            
            self.qsca_el[ : , n ] =  ( 2 * np.pi / ( ( k )**2 ) * ( 2 * n + 1 ) ) * np.abs( an )**2
            self.qsca_ma[ : , n ] =  ( 2 * np.pi / ( ( k )**2 ) * ( 2 * n + 1 ) ) * np.abs( bn )**2
       
            self.qext_el[ : , n ] =  ( 2 * np.pi / ( ( k )**2 ) * ( 2 * n + 1 ) ) * np.real( an )
            self.qext_ma[ : , n ] =  ( 2 * np.pi / ( ( k )**2 ) * ( 2 * n + 1 ) ) * np.real( bn )
            
            self.qabs_el[ : , n ] =  self.qext_el[:,n] - self.qsca_el[:,n]
            self.qabs_ma[ : , n ] =  self.qext_ma[:,n] - self.qsca_ma[:,n]
            
                        
        self.qsca = np.sum( self.qsca_el , 1 ) + np.sum( self.qsca_ma , 1 )
        self.qext = np.sum( self.qext_el , 1 ) + np.sum( self.qext_ma , 1 )
        self.qabs = self.qext - self.qsca
    
# Define the special functions needed for calculation of eigenmodes of the Core-Shell spheroid
    def __Psi( self , z , n ):
        return z * self.__Jspher( n , z )
    
    def __DPsi( self , z , n ):
        return z *  self.__Jspher( n - 1 , z ) - n *  self.__Jspher( n , z )
    
    def __Chi( self , z , n ):
        return z * self.__Yspher( n , z )
    
    def __DChi( self , z , n ):
        return z * self.__Yspher( n - 1 , z ) - n * self.__Yspher( n , z )
    
    def __Xi( self , z , n ):
        return z *  self.__Hspher( n , z )
        
    def __DXi( self , z , n ):
        return z *  self.__Hspher( n - 1 , z ) - n *  self.__Hspher( n , z )
        
    def __Jspher( self , n , z ):
        return np.sqrt( 0.5 * np.pi / z ) * bessel(  n + 0.5 , z )
        
    def __Yspher( self , n , z ):
        return np.sqrt( 0.5 * np.pi / z ) * neumann( n + 0.5 , z )
        
    def __Hspher( self , n , z ):
        return np.sqrt( 0.5 * np.pi / z ) *  hankel( n + 0.5 , z )
          
