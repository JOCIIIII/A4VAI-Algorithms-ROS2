############################################################
#
#   - Name : utility_funcs.py
#
#                   -   Created by E. T. Jeong, 2023.01.09
#
############################################################

#.. Library
# pulbic libs.
from math import sin, cos, atan2, sqrt, asin
from numpy import zeros

# private libs.

#.. get DCM from Euler angle
def DCM_from_euler_angle( EulerAng ):

    # Local Variable 
    spsi            =   sin( EulerAng[2] )
    cpsi            =   cos( EulerAng[2] )
    sthe            =   sin( EulerAng[1] )
    cthe            =   cos( EulerAng[1] )
    sphi            =   sin( EulerAng[0] )
    cphi            =   cos( EulerAng[0] )
    
    # DCM from 1-frame to 2-frame
    c1_2            =   zeros((3,3))
    
    c1_2[0,0]       =   cpsi * cthe 
    c1_2[1,0]       =   cpsi * sthe * sphi - spsi * cphi 
    c1_2[2,0]       =   cpsi * sthe * cphi + spsi * sphi 
    
    c1_2[0,1]       =   spsi * cthe 
    c1_2[1,1]       =   spsi * sthe * sphi + cpsi * cphi 
    c1_2[2,1]       =   spsi * sthe * cphi - cpsi * sphi 
    
    c1_2[0,2]       =   -sthe 
    c1_2[1,2]       =   cthe * sphi 
    c1_2[2,2]       =   cthe * cphi 

    return c1_2

#.. get azimuth & elevation angle from vector
def azim_elev_from_vec3(Vec3):
    azim    =   atan2(Vec3[1],Vec3[0])
    elev    =   atan2(-Vec3[2], sqrt(Vec3[0]*Vec3[0]+Vec3[1]*Vec3[1]))
    return azim, elev

#.. set_angle_range
def set_angle_range(ang_in):
    return atan2(sin(ang_in),cos(ang_in))

#.. Euler to Quaternion
def Euler2Quaternion(EulerAng):

    Roll  = EulerAng[0]
    Pitch = EulerAng[1]
    Yaw   = EulerAng[2]

    CosYaw = cos(Yaw * 0.5)
    SinYaw = sin(Yaw * 0.5)
    CosPitch = cos(Pitch * 0.5)
    SinPitch = sin(Pitch * 0.5)
    CosRoll = cos(Roll * 0.5)
    SinRoll= sin(Roll * 0.5)
    
    w = CosRoll * CosPitch * CosYaw + SinRoll * SinPitch * SinYaw
    x = SinRoll * CosPitch * CosYaw - CosRoll * SinPitch * SinYaw
    y = CosRoll * SinPitch * CosYaw + SinRoll * CosPitch * SinYaw
    z = CosRoll * CosPitch * SinYaw - SinRoll * SinPitch * CosYaw
    
    return w, x, y, z

#.. Quaternion to Euler
def Quaternion2Euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    Roll = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Pitch = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Yaw = atan2(t3, t4)

    return Roll, Pitch, Yaw