import random, sys, linecache, csv, math, time
import numpy as np
from scipy.spatial import KDTree

start_time = time.time()
########## Write the neighboring list between two chosen atoms A and B

trans = 8.4

atomA = 'H'   # Input atom A
atomB = 'O'   # Input atom B

rvw_a = 1.2   # van der waals radius of atom A (H)
rvw_b = 1.53  # van der Waals radius of atom B (O)
rvw_err = 0.1

#limitdist = 2.0   # Given the upper limit of the neighboring distance

nframe = 18

output = open('dist_%.1f' %(trans),'w')
#output = open('dist_%d' %(trans),'w')

for index in range(nframe):
  rangle = index*20
  print (rangle)

  filename = './t%.1f_opt/t%.1f_%d' %(trans,trans,rangle) 
  #filename = './t%d_opt/t%d_%d' %(trans,trans,rangle) 

  #output = open('%s%s' %(atomA, atomB),'w')
  #output = open('neighbor_PS4','w')


  contcar = open(filename, 'r')

  system = contcar.readline()
  scale = float(contcar.readline().rstrip('\n'))
  #print scale

  #get lattice vectors
  a1 = np.array([ float(s)*scale for s in contcar.readline().rstrip('\n').split() ])
  a2 = np.array([ float(s)*scale for s in contcar.readline().rstrip('\n').split() ])
  a3 = np.array([ float(s)*scale for s in contcar.readline().rstrip('\n').split() ])

  print (a1)
  print (a2)
  print (a3)

  #Read contcar
  element_names = contcar.readline().rstrip('\n').split()

  element_dict = {}
  element_numbers = contcar.readline().rstrip('\n').split()

  element_start = {}

  i = 0
  N = 0
  for t in range(len(element_names)):
    element_start[element_names[t]] = N+1
    element_dict[element_names[t]] = int(element_numbers[i])
    N += int(element_numbers[i])
    
    #print N
    i += 1

  print (element_dict)
  print (element_start)
  print (element_dict[atomA])
  contcar.close()

  count = 0
  line_start = 8
  coord_a = np.empty(shape=[0,3])
  coord_b = np.empty(shape=[0,3])

  for i in range(element_dict[atomA]):
    line = line_start + element_start[atomA]+i
    ra = linecache.getline(filename,line).rstrip('\n').split()[0:]
    #print ra
    [xa, ya, za] = [float(s) for s in ra]
    if xa < 0.0 or xa > 1.0:
      xa=xa-1.0*np.sign(xa)
    if ya < 0.0 or ya > 1.0:
      ya=ya-1.0*np.sign(ya)
    if za < 0.0 or za > 1.0:
      za=za-1.0*np.sign(za)
            
    xxa = xa*a1[0] + ya*a2[0] + za*a3[0]
    yya = xa*a1[1] + ya*a2[1] + za*a3[1]
    zza = xa*a1[2] + ya*a2[2] + za*a3[2]

    coord_a = np.append(coord_a,[[xxa,yya,zza]],axis=0)
  
  for i in range(element_dict[atomB]):
    line = line_start + element_start[atomB]+i
    rb = linecache.getline(filename,line).rstrip('\n').split()[0:]
    #print ra
    [xb, yb, zb] = [float(s) for s in rb]
    if xb < 0.0 or xb > 1.0:
      xb=xb-1.0*np.sign(xb)
    if yb < 0.0 or yb > 1.0:
      yb=yb-1.0*np.sign(yb)
    if zb < 0.0 or zb > 1.0:
      zb=zb-1.0*np.sign(zb)
            
    xxb = xb*a1[0] + yb*a2[0] + zb*a3[0]
    yyb = xb*a1[1] + yb*a2[1] + zb*a3[1]
    zzb = xb*a1[2] + yb*a2[2] + zb*a3[2]

    coord_b = np.append(coord_b,[[xxb,yyb,zzb]],axis=0)


  sort_coord_a = coord_a[coord_a[:, 2].argsort()]
  upper_a = int(len(sort_coord_a)/2)
  upper_coord_a = sort_coord_a[:upper_a]
  lower_coord_a = sort_coord_a[upper_a:]

  sort_coord_b = coord_b[coord_b[:, 2].argsort()]
  upper_b = int(len(sort_coord_b)/2)
  upper_coord_b = sort_coord_b[:upper_b]
  lower_coord_b = sort_coord_b[upper_b:]


  #print (upper_coord_a)
  
  aa_dist = []
  ab_dist = []
  radi_aa = []
  radi_ab = []
  count_aa = 0
  count_ab = 0
  for j in range(len(upper_coord_a)):
    point_a = upper_coord_a[j]
    #print(point_a) 

    kdtree_aa = KDTree(lower_coord_a)
    dist_aa, index_aa = kdtree_aa.query(point_a)
    if 2 < dist_aa <= 2*rvw_a+rvw_err:
      count_aa += 1
      radi_aa.append(dist_aa)
    aa_dist.append(dist_aa)

    kdtree_ab = KDTree(lower_coord_b)
    dist_ab, index_ab = kdtree_ab.query(point_a)
    if 1 < dist_ab <= rvw_a+rvw_b+rvw_err:
      count_ab += 1
      radi_ab.append(dist_ab)

    ab_dist.append(dist_ab)
  
  ba_dist = []
  bb_dist = []
  radi_ba = []
  radi_bb = []
  count_ba = 0
  count_bb = 0
  for k in range(len(upper_coord_b)):
    point_b = upper_coord_b[k]

    kdtree_ba = KDTree(lower_coord_a)
    dist_ba, index_ba = kdtree_ba.query(point_b)
    if 1 < dist_ba <= rvw_a+rvw_b+rvw_err:
      count_ba += 1
      radi_ba.append(dist_ba)

    ba_dist.append(dist_ba)

    kdtree_bb = KDTree(lower_coord_b)
    dist_bb, index_bb = kdtree_bb.query(point_b)
    if dist_bb <= 2*rvw_b+rvw_err:
      count_bb += 1
      radi_bb.append(dist_bb)

    bb_dist.append(dist_bb)

  min_aa = min(aa_dist)
  min_ab = min(ab_dist)
  min_bb = min(bb_dist)
  min_ba = min(ba_dist)
  
  ave_radi_aa = sum(radi_aa)/count_aa if count_aa != 0 else 0
  ave_radi_ab = sum(radi_ab)/count_ab if count_ab != 0 else 0
  ave_radi_bb = sum(radi_bb)/count_bb if count_bb != 0 else 0
  ave_radi_ba = sum(radi_ba)/count_ba if count_ba != 0 else 0
  

  print(min_aa, min_ab, min_bb, min_ba, ave_radi_aa, ave_radi_ab, ave_radi_bb, ave_radi_ba)
  output.write("%g %6.4g %6.4g %6.4g %6.4g %6.4g %6.4g %6.4g %6.4g %6.4g %5.4g %5.4g %5.4g %5.4g\n"%(trans,rangle,min_aa,min_ab,min_bb,min_ba,ave_radi_aa, ave_radi_ab, ave_radi_bb, ave_radi_ba, count_aa, count_ab,count_bb,count_ba))

output.close()
print ("Time elapsed: ", time.time() - start_time, "s")  


