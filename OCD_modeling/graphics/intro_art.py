# Code for creating mesh art for intro figure
# Note that it runs on the "base" virtual env and not the "bayesenv"
# Author: Sebastien Naze 
# QIMR Berghofer 2022-2023

import joblib 
from joblib import delayed, Parallel
import matplotlib
from matplotlib import pyplot as plt
import nilearn
import numpy as np
import os
import pyvista as pv
import pymeshfix as mf
import sys
from time import sleep

proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
atlas_dir = os.path.join(proj_dir, 'utils')
fs_dir = '/usr/local/freesurfer/'

sys.path.insert(0, os.path.join(code_dir, 'utils'))
from OCD_baseline.utils import atlaser

sys.path.insert(0, os.path.join(code_dir, 'graphics'))
import baseline_visuals


#subcortex = ['Right_Hippocampus', 'Right_Amygdala', 'Right_PosteriorThalamus', 'Right_AnteriorThalamus', 'Right_NucleusAccumbens', 'Right_GlobusPallidus', 'Right_Putamen', 'Right_Caudate']
#           'Left_Hippocampus', 'Left_Amygdala', 'Left_PosteriorThalamus', 'Left_AnteriorThalamus', 'Left_NucleusAccumbens', 'Left_GlobusPallidus', 'Left_Putamen', 'Left_Caudate']

subcortex = ['Right_NucleusAccumbens', 'Right_Putamen']

opts = {'Putamen': {'color': 'RoyalBlue',
                          'show_edges': False,
                          'line_width': 1,
                          'opacity':1},
        'NucleusAccumbens': {   'color': 'firebrick',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'Hippocampus': {'color': 'orange',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity': 0.2,
                          'style': 'wireframe'},
        'Amygdala': {'color': 'purple',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'GlobusPallidus': {'color': 'gray',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'AnteriorThalamus': {'color': 'black',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'PosteriorThalamus': {'color': 'black',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'Caudate': {'color': 'green',
                          'show_edges': True,
                          'line_width': 0.5,
                          'opacity':0.5,
                          'style': 'wireframe'},
        'cortex' : {  'color': 'aliceblue',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 0.6 },
        'VisCent' : {  'color': 'bisque',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'VisPeri' : {  'color': 'tan',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'SomMot' : {  'color': 'palegreen',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'DorsAttn': {  'color': 'paleturquoise',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'VentAttn': {  'color': 'plum',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'Limbic': {  'color': 'lightpink',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'Cont': {  'color': 'beige',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'Default': {  'color': 'gold',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 },
        'TempPar' : {  'color': 'lavender',
                    'show_edges': False,
                    'line_width': 1,
                    'opacity': 1 }       
        }

# Figure 1: Single fronto striatal pathway
"""
opts = {'Putamen': {'color': 'limegreen',
                          'show_edges': False,
                          'line_width': 1,
                          'opacity':1},
        'Caudate': {'color': 'limegreen',
                          'show_edges': False,
                          'line_width': 1,
                          'opacity':1},
        'NucleusAccumbens': {   'color': 'limegreen',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'OFC': {   'color': 'dodgerblue',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'PFC': {   'color': 'dodgerblue',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'cortex_mesh' : {   'color': 'gray',
                            'show_edges': True,
                            'line_width': 0.5,
                            'opacity': 0.2,
                            'style': 'wireframe' 
                        } ,
        'cortex_surf' : {   'color': 'whitesmoke',
                            'show_edges': False,
                            'line_width': 0.1,
                            'opacity': 1,
                            'style': 'surface' 
                   } 
        }

"""
# Option 1: opaque surfaces

opts = {'Putamen': {'color': 'magenta',
                          'show_edges': False,
                          'line_width': 1,
                          'opacity':1},
        'NucleusAccumbens': {   'color': 'red',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'OFC': {   'color': 'blue',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'PFCl_2': {   'color': 'green',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'cortex_mesh' : {   'color': 'gray',
                            'show_edges': True,
                            'line_width': 0.5,
                            'opacity': 0.2,
                            'style': 'wireframe' 
                        } ,
        'cortex_surf' : {   'color': 'lightgray',
                            'show_edges': False,
                            'line_width': 0.1,
                            'opacity': 1,
                            'style': 'surface' 
                   } 
        }

# Figure 2: Two-pathways fronto-striatal system
# Otion 2: mesh surfaces 
"""
opts = {'Putamen': {'color': 'orange', #'lightcoral'
                          'show_edges': True,
                          'line_width': 0.2,
                          'opacity':0.2,
                          'style': 'wireframe' },
        'NucleusAccumbens': {   'color': 'red',
                                      'show_edges': True,
                                      'line_width': 0.2,
                                      'opacity':0.2,
                                      'style': 'wireframe' },
        'OFC': {   'color': 'blue',
                                      'show_edges': True,
                                      'line_width': 0.2,
                                      'opacity':0.2,
                                      'style': 'wireframe' },
        'PFCl_2': {   'color': 'lightskyblue',
                                      'show_edges': True,
                                      'line_width': 0.2,
                                      'opacity':0.2,
                                      'style': 'wireframe' },
        'cortex_mesh' : {   'color': 'gray',
                            'show_edges': True,
                            'line_width': 0.5,
                            'opacity': 0.2,
                            'style': 'wireframe' 
                        } ,
        'cortex_surf' : {   'color': 'gray',
                            'show_edges': False,
                            'line_width': 0.1,
                            'opacity': 1,
                            'style': 'surface' 
                   } 
        }
"""

def create_mesh(roi_img, alpha=1., n_iter=80):
    """ creates pyvista mesh from all non-zero entries of a niftii image """
    # create ROI point cloud
    ijk = np.array(np.where(roi_img.get_fdata()>0))
    xyz_coords = nilearn.image.coord_transform(ijk[0], ijk[1], ijk[2], affine=roi_img.affine)
    #roi_points = pv.PointSet(np.array(xyz_coords))
    roi = pv.PolyData(np.array(xyz_coords).T)

    # extract mesh from point cloud
    mesh = roi.delaunay_3d(alpha=alpha).extract_geometry().clean()

    # repair mesh for inconsistencies
    mesh = mf.MeshFix(mesh)
    mesh.repair()
    mesh = mesh.mesh

    # smooth mesh (use Taubin to keep volume)
    mesh.smooth_taubin(n_iter=n_iter, inplace=True)
    return mesh


def create_region_mesh(region):
    roi_img = atlazer.create_subatlas_img(region)
    mesh = create_mesh(roi_img)
    return mesh

# Additionally, all the modules other than ipygany and pythreejs require a framebuffer, which can be setup on a headless environment with pyvista.start_xvfb().
pv.start_xvfb()

atlazer = atlaser.Atlaser('schaefer100_tianS1')

pl = pv.Plotter(window_size=[800,600], notebook=True)
pl.background_color='white'

meshes = dict()

# subcortex
for region in subcortex:
    # get nifti img from region(s)
    roi_img = atlazer.create_subatlas_img(region)
    mesh = create_mesh(roi_img)
    meshes[region] = mesh
    
# cortex
mshs = Parallel(n_jobs=32)(delayed(create_region_mesh)(region) for region in atlazer.node_names[58:108]) # use :400 for bilateral
for region,mesh in zip(atlazer.node_names[58:108], mshs):
    meshes[region] = mesh
    
# plot
for region,mesh in meshes.items():
    roi = [key for key in opts.keys() if key in region]
    roi='cortex_mesh' if roi==[] else roi[0]
    pl.add_mesh(mesh, **opts[roi])

pl.camera_position = 'yz'
pl.camera.roll += 10
pl.camera.zoom(1.0)
#pl.camera.clipping_range = (0,9999)
#pl.save_graphic(filename='/home/sebastin/working/lab_lucac/sebastiN/projects/OCD_modeling/img/frontostriatal_003_20230905.pdf',
#                raster=True, painter=False)
"""
pl.camera.azimuth -= 45
pl.renderer.reset_camera_clipping_range()
pl.screenshot(filename='/home/sebastin/working/lab_lucac/sebastiN/projects/OCD_modeling/img/frontostriatal_lateralposterior001_20230906.png',
              transparent_background=True,
              return_img=False,
              )

pl.camera.azimuth += 45
pl.renderer.reset_camera_clipping_range()
pl.screenshot(filename='/home/sebastin/working/lab_lucac/sebastiN/projects/OCD_modeling/img/frontostriatal_lateralanterior001_20230906.png',
              transparent_background=True,
              return_img=False,
              )

pl.camera.azimuth += 225
pl.renderer.reset_camera_clipping_range()
pl.screenshot(filename='/home/sebastin/working/lab_lucac/sebastiN/projects/OCD_modeling/img/frontostriatal_medialposterior001_20230906.png',
              transparent_background=True,
              return_img=False,
              )
"""
pl.camera.azimuth += 180
pl.renderer.reset_camera_clipping_range()
pl.screenshot(filename='/home/sebastin/working/lab_lucac/sebastiN/projects/OCD_modeling/img/OFC_PFC_NAcc_dPut_medial002_20231129.png',
              transparent_background=True,
              return_img=False,
              )

pl.show(jupyter_backend='panel')
