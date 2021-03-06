import bpy
import os
from math import *
from mathutils import *
import pathlib

import numpy as np


def look_at(obj_camera, point):
    loc_camera = obj_camera.location#obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
res = 128
bpy.context.scene.render.resolution_x = res
bpy.context.scene.render.resolution_y = res


print(pathlib.Path(__file__).parent.resolve())
#set your own target here
target = bpy.data.objects['horse_statue_01']
cam = bpy.data.objects['Camera']
t_loc_x = target.location.x
t_loc_y = target.location.y
cam_loc_x = cam.location.x
cam_loc_y = cam.location.y


context = bpy.context
scene = context.scene
camera = scene.camera  # bpy.types.Camera
render = scene.render

# compute projection matrix -> Returns: 4x4 projection matrix


# The different radii range
radius_range = range(3,9)

R = (target.location.xy-cam.location.xy).length # Radius
init_angle  = (1-2*bool((cam_loc_y-t_loc_y)<0))*acos((cam_loc_x-t_loc_x)/R)-2*pi*bool((cam_loc_y-t_loc_y)<0) # 8.13 degrees
target_angle = (pi*2 - init_angle) # Go 360-8 deg more
num_steps = 15 #how many rotation steps
r = 4
i = 0
theta = np.linspace(0, np.pi/2-0.03, 15)

transMat = np.zeros((len(theta) * num_steps - num_steps+1,4,4))
pwd = str(pathlib.Path(__file__).parent.resolve())
for z in theta:
    cam.location.z = r*cos(z)
    for x in range(1,num_steps+1):
        alpha = init_angle + (x)*target_angle/num_steps
        cam.location.x = t_loc_x+cos(alpha) * r * sin(z)
        cam.location.y = t_loc_y+sin(alpha) * r * sin(z)
        print(i)
        # Define SAVEPATH and output filename
        look_at(cam, target.matrix_world.to_translation());
        file = os.path.join(pwd, f'../../data/{res}/{i}')#+str(0)+'_'+ str(r)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))

        projection_mat = camera.matrix_basis
        # print(np.asarray(projection_mat))
        transMat[i] = np.asarray(projection_mat)
        i += 1

        # Render
        bpy.context.scene.render.filepath = file
        bpy.ops.render.render(write_still=True)
        if i == 1: break
# print('\n\n', transMat)
# print(len(theta) * num_steps - num_steps+1, len(transMat))
with open(os.path.join(pwd[:-15], f'../data/posEnc_{res}.npz'), 'wb') as f:
    np.save(f,transMat)
i = 0
res = 256
bpy.context.scene.render.resolution_x = res
bpy.context.scene.render.resolution_y = res

for z in theta:
    cam.location.z = r*cos(z)
    for x in range(1,num_steps+1):
        alpha = init_angle + (x)*target_angle/num_steps
        # cam.rotation_euler[2] = pi/2 + alpha #
        cam.location.x = t_loc_x+cos(alpha) * r * sin(z)
        cam.location.y = t_loc_y+sin(alpha) * r * sin(z)
        print(i)
        # Define SAVEPATH and output filename
        look_at(cam, target.matrix_world.to_translation());
        file = os.path.join(pwd, f'../../data/{res}/{i}')#+str(0)+'_'+ str(r)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))
        projection_mat = camera.matrix_basis

        # print(np.asarray(projection_mat))
        transMat[i] = np.asarray(projection_mat)
        i += 1
        # Render
        bpy.context.scene.render.filepath = file
        bpy.ops.render.render(write_still=True)
        if i == 1: break

with open(os.path.join(pwd[:-15], f'../data/posEnc_256.npz'), 'wb') as f:
    np.save(f,transMat)
res = 512
bpy.context.scene.render.resolution_x = res
bpy.context.scene.render.resolution_y = res


i = 0

# for z in theta:
#     cam.location.z = r*cos(z)
#     for x in range(1,num_steps+1):
#         alpha = init_angle + (x)*target_angle/num_steps
#         # cam.rotation_euler[2] = pi/2 + alpha #
#         cam.location.x = t_loc_x+cos(alpha) * r * sin(z)
#         cam.location.y = t_loc_y+sin(alpha) * r * sin(z)
#         print(i)
#         # Define SAVEPATH and output filename
#         look_at(cam, target.matrix_world.to_translation());
#         file = os.path.join(pwd, f'../../data/{res}/{i}')#+str(0)+'_'+ str(r)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))
#         projection_mat = camera.matrix_basis

#         # print(np.asarray(projection_mat))
#         transMat[i] = np.asarray(projection_mat)
#         i += 1
#         # Render
#         bpy.context.scene.render.filepath = file
#         bpy.ops.render.render(write_still=True)
#         if i == 1: break



# with open(os.path.join(pwd[:-15], f'../data/posEnc_512.npz'), 'wb') as f:
#     np.save(f,transMat)
# with open(os.path.join(pwd[:-15], f'../data/posEnc_512.npz'), 'rb') as f:
#     transMat = np.load(f)
#     print(transMat)



