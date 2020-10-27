import numpy as np # linear algebra
import xml.etree.ElementTree as ET # for parsing XML
import matplotlib.pyplot as plt # to show images
from PIL import Image # to read images
import os
import glob

root_images = r'/home/nthds/Documents/Solutions/Data_Aug/synthetic-images/jpg_images'
root_annots = r'/home/nthds/Documents/Solutions/Data_Aug/synthetic-images/jpg_annotations'
# res_dir = r'/home/nthds/Documents/Solutions/Data_Aug/synthetic-images/results_imgs'

all_images=os.listdir(root_images)
print(f"Total images : {len(all_images)}")

breeds = glob.glob(root_annots)
annotation=[]
for b in breeds:
	annotation+=glob.glob(b+"/*")
print(f"Total annotation : {len(annotation)}")

breed_map={}
for annot in annotation:
	breed=annot.split("/")[-2]
	index=breed.split("-")[0]
	breed_map.setdefault(index,breed)
	
print(f"Total Breeds : {len(breed_map)}")

def bounding_box(image):
	#bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])
	#print (bpath)
	#print(root_annots)
	#print (str(breed_map[image.split("_")[0]]))
	#print (str(image.split(".")[0]))
	bpath=root_annots+"/"+str(image.split(".")[0]+".xml")
	tree = ET.parse(bpath)
	root = tree.getroot()
	objects = root.findall('object')
	bbox_list=[]
	for o in objects:
		bndbox = o.find('bndbox') # reading bound box
		bndname = o.find("name").text # reading bound box

		xmin = int(bndbox.find('xmin').text)
		ymin = int(bndbox.find('ymin').text)
		xmax = int(bndbox.find('xmax').text)
		ymax = int(bndbox.find('ymax').text)
		bbox_list.append([xmin,ymin,xmax,ymax, bndname])
	# print(bbox)
	# exit()  
	# return (xmin,ymin,xmax,ymax, bndname)
	return bbox_list	
# def get_bounding_boxes():


# lt.figure(figsize=(10,10))
# bbox=[]
for i,image in enumerate(all_images):
	try:
		print("image", image)
		bbox_list = bounding_box(image)
		# print(bbox_list)
		# exit()
		i += 1
		try:
			for boundary_box in bbox_list:

				bbox = boundary_box[0:-1]
			# print("bbox" ,bbox)
			# exit()
				bname = boundary_box[-1]
			# print("bname", bname)
			# exit()
				im = Image.open(os.path.join(root_images,image))
				im=im.crop(bbox)
				  
				im.save('./results_imgs/{}.jpeg'.format(bname + "_" + str(i),im)) 
				i += 1
		except Exception as e:
		  print(e)
	except Exception as e:
		print(e)