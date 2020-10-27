import os
import json
import argparse
import numpy as np
import random
import math
from PIL import Image, ImageEnhance
import xml.etree.ElementTree as et
import shutil
import re

root_path = os.getcwd()
# Entrypoint Args
parser = argparse.ArgumentParser(description='Create synthetic training data for object detection algorithms.')
parser.add_argument("-bkg", "--backgrounds", type=str, default="Backgrounds/",
                                        help="Path to background images folder.")
parser.add_argument("-obj", "--objects", type=str, default="Objects/",
                                        help="Path to object images folder.")
parser.add_argument("-o", "--output", type=str, default="TrainingImages/",
                                        help="Path to output images folder.")
parser.add_argument("-ann", "--annotate", type=bool, default=True,
                                        help="Include annotations in the data augmentation steps?")
parser.add_argument("-s", "--sframe", type=bool, default=False,
                                        help="Convert dataset to an sframe?")
parser.add_argument("-g", "--groups", type=bool, default=True,
                                        help="Include groups of objects in training set?")
parser.add_argument("-mut", "--mutate", type=bool, default=True,
                                        help="Perform mutations to objects (rotation, brightness, sharpness, contrast)")
args = parser.parse_args()


# Prepare data creation pipeline
base_bkgs_path = args.backgrounds
## bkg_images = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")]
bkg_images = [f for f in os.listdir(base_bkgs_path) if f.endswith(".jpg")]
objs_path = args.objects
obj_images = [f for f in os.listdir(objs_path) if not f.startswith(".")]
# print(obj_images)
# exit()
sizes = [0.4, 0.6, 0.8, 1, 1.2] # different obj sizes to use TODO make configurable
# sizes = [1, 2, 3, 5, 7] # different obj sizes to use TODO make configurable
count_per_size = 4 # number of locations for each obj size TODO make configurable
annotations = [] # store annots here
output_images = args.output
n = 1
count_per_image =5


                        # obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, count=count_per_size)            
# Helper functions
def get_obj_positions(obj, bkg, count=1):
        obj_w, obj_h = [], []
        x_positions, y_positions = [], []
        bkg_w, bkg_h = bkg.size
        # Rescale our obj to have a couple different sizes
        obj_sizes = [tuple([int(s*x) for x in obj.size]) for s in sizes]
        for w, h in obj_sizes:
                obj_w.extend([w]*count)
                obj_h.extend([h]*count)
                x_positions.extend(list(np.random.randint(0, max_x, count)))
                y_positions.extend(list(np.random.randint(0, max_y, count)))

        return obj_h, obj_w, x_positions, y_positions


def get_box(obj_w, obj_h, max_x, max_y):
        x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
        x2, y2 = x1 + obj_w, y1 + obj_h
        return [x1[0], y1[0], x2[0], y2[0]]


def is_in(x, y, box):
        x1, y1, x2, y2 = box
        return not (x < x1 or x > x2 or y < y1 or y > y2)

# check if two boxes intersect
def intersects(box, new_box):
        box_x1, box_y1, box_x2, box_y2 = box
        x1, y1, x2, y2 = new_box
        # check if box overlaps new_box
        if is_in(box_x1, box_y1, new_box) or is_in(box_x1, box_y2, new_box) or is_in(box_x2, box_y1, new_box) or is_in(box_x2, box_y2, new_box):
                return True
        # check if new_box overlaps box
        if is_in(x1, y1, box) or is_in(x1, y2, box) or is_in(x2, y1, box) or is_in(x2, y2, box):
                return True
        return False
  
def exceeds_bounds(box, bkg):
        bkg_w, bkg_h = bkg.size
        x1, y1, x2, y2, = box

        return (x2 > bkg_w or y2 > bkg_h)

def find_existing_boxes_from_xml(bkg_path):
    boxes = []  
    # Load the background image
    bkg_path = base_bkgs_path + bkg
    print(bkg_path)
    bkg_xml_path_one = os.path.splitext(bkg_path.split("/")[1].split(".xml")[0])[0] + ".xml"
    bkg_xml_path = os.path.join(os.path.join(root_path, "Backgrounds"),bkg_xml_path_one) 
    tree = et.parse(bkg_xml_path)
    root = tree.getroot()
    for object in root.iter("object"):
        name = object.find("name").text
        if name in ('Invoice_number', 'Invoice_date', 'Line_item',
                     'Line_item_header', 'Factoring', 'Trucking',
                     'Reference', 'Total'):
            xmin = object.find("bndbox").find("xmin")
            ymin = object.find("bndbox").find("ymin")
            xmax = object.find("bndbox").find("xmax")
            ymax = object.find("bndbox").find("ymax")
            new_xmin = int(xmin.text)
            new_xmax = int(xmax.text)
            new_ymin = int(ymin.text)
            new_ymax = int(ymax.text)
            # print(new_xmin, new_ymin, new_xmax, new_ymax)
            # exit()
            boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    return boxes

def get_group_obj_positions(obj_group, bkg, existing_boxes):
        bkg_w, bkg_h = bkg.size
        # boxes needs to have parsed xml file boxes first so that it doesnt intersect
        # boxes = []
        # print("#####", boxes)
        # exit()
        print("existing boxes", existing_boxes)
        boxes = []

        objs = [Image.open(objs_path + obj_images[i]) for i in obj_group]
        obj_sizes = [tuple([int(0.6*x) for x in i.size]) for i in objs]
        for w, h in obj_sizes:
                # set background image boundaries
                max_x, max_y = abs(bkg_w-w), abs(bkg_h-h)
                # print("MAXIMUM X ", max_x, "MAXIMUM Y ", max_y)

                # get new box coordinates for the obj on the bkg
                invalid = True
                while invalid:
                        invalid = False
                        new_box = get_box(w, h, max_x, max_y)
                        for box in existing_boxes:
                            if intersects(box, new_box):
                                invalid = True
                                break
                        if invalid:
                            continue
                        for box in boxes:
                            if intersects(box, new_box):
                                invalid = True
                                break
                # append our new box
                boxes.append(new_box)
        print("boxes", boxes)

        # exit()
        return obj_sizes, boxes
        
def mutate_image(img):
        # resize image for random value
        resize_rate = random.choice(sizes)
        img = img.resize([int(img.width*resize_rate), int(img.height*resize_rate)], Image.BILINEAR)

        # rotate image for random andle and generate exclusion mask 
        rotate_angle = random.randint(0,360)
        mask = Image.new('L', img.size, 255)
        img = img.rotate(rotate_angle, expand=True)
        mask = mask.rotate(rotate_angle, expand=True)

        # perform some enhancements on image
        enhancers = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
        enhancers_count = random.randint(0,3)
        for i in range(0,enhancers_count):
                enhancer = random.choice(enhancers)
                enhancers.remove(enhancer)
                img = enhancer(img).enhance(random.uniform(0.5,1.5))

        return img, mask

def add_xml(tree, obj_xmin, obj_ymin, obj_xmax, obj_ymax, final_label):
        root = tree.getroot()
        name = final_label
        Object = et.SubElement(root, "object")
        Name = et.SubElement(Object, "name")
        Name.text = name
        Pose = et.SubElement(Object, "pose")
        Pose.text = "Unspecified"
        Truncated = et.SubElement(Object, "truncated")
        Truncated.text = "0"
        Difficult = et.SubElement(Object, "difficult")
        Difficult.text = "0"
        Bndbox = et.SubElement(Object, "bndbox")
        Xmin = et.SubElement(Bndbox, "xmin")
        Xmin.text = str(obj_xmin)
        Ymin = et.SubElement(Bndbox, "ymin")
        Ymin.text = str(obj_ymin)
        Xmax = et.SubElement(Bndbox, "xmax")
        Xmax.text = str(obj_xmax)
        Ymax = et.SubElement(Bndbox, "ymax")
        Ymax.text = str(obj_ymax)
# obj_xmin, obj_ymin, obj_xmax, obj_ymax, final_label
        tree = et.ElementTree(root)
        # tree.write(xml_file)
# obj_xmin, obj_ymin, obj_xmax, obj_ymax
        return tree
        # exit()

                                # output_fp = output_images + str(n) + ".png"
        

if __name__ == "__main__":
    try:
        # Make synthetic training data
        print("Making synthetic images.", flush=True)
        for bkg in bkg_images:
            try:
                # Load the background image
                bkg_path = base_bkgs_path + bkg
                bkg_img = Image.open(bkg_path)
                bkg_x, bkg_y = bkg_img.size
######### single objects ##################
                # # Do single objs first
                # for i in obj_images:
                #     # Load the single obj
                #     i_path = objs_path + i
                #     obj_img = Image.open(i_path)

                #     # Get an array of random obj positions (from top-left corner)
                #     obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, count=count_per_size)            
                #     # print(obj_h, obj_w, x_pos, y_pos)
                #     # exit()
                        
                        

                #     # Create synthetic images based on positions
                #     for h, w, x, y in zip(obj_h, obj_w, x_pos, y_pos):
                #         # Copy background
                #         bkg_w_obj = bkg_img.copy()
                                
                #         if args.mutate:
                #             new_obj, mask = mutate_image(obj_img)
                #             # Paste on the obj
                #             bkg_w_obj.paste(new_obj, (x, y), mask)
                #         else:
                #             # Adjust obj size
                #             new_obj = obj_img.resize(size=(w, h))
                #             # Paste on the obj
                #             bkg_w_obj.paste(new_obj, (x, y))
                #         output_fp = output_images + str(n) + ".png"
                #         # Save the image
                #         bkg_w_obj.save(fp=output_fp, format="png")

                #         if args.annotate:
                #             # Make annotation
                #             ann = [{'coordinates': {'height': h, 'width': w, 'x': x+(0.5*w), 'y': y+(0.5*h)}, 'label': i.split(".png")[0]}]
                #             # Save the annotation data
                #             annotations.append({
                #                 "path": output_fp,
                #                 "annotations": ann
                #             })
                #         #print(n)
                #         n += 1
######### single objects ##################

                if args.groups:
                    try:
                        # create 4 synthetic images from each input image
                        for j in range(count_per_image):
                            try:
                                # create 2 to 4 random objects
                                group = np.random.randint(0, len(obj_images) -1, np.random.randint(2, 5, 1))
                                print("############", group)
                                # Get sizes and positions
                                ann = []
                                existing_boxes = find_existing_boxes_from_xml(bkg_path)
                                try:
                                    obj_sizes, boxes = get_group_obj_positions(group, bkg_img, existing_boxes)
                                except Exception as e:
                                    print(e)   
                                bkg_w_obj = bkg_img.copy()

                                # show existing boxes in black
                                # for box in existing_boxes:
                                #    bkg_w_obj.paste(0, tuple(box))
                                xml_file = os.path.join(os.path.join(root_path, "Backgrounds"), bkg[:-3] + "xml")
                                xml_data = et.parse(xml_file)

                                # For each obj in the group
                                for i, size, box in zip(group, obj_sizes, boxes):
                                        # Get the obj
                                        obj = Image.open(objs_path + obj_images[i])
                                        obj_w, obj_h = size
                                        x_pos, y_pos = box[:2]
                                        # Resize it as needed
                                        x_scale = 1
                                        y_scale = 1
                                        if x_pos + obj_w > bkg_x:
                                                x_scale = float(bkg_x - x_pos) / obj_w
                                        if y_pos + obj_h > bkg_y:
                                                y_scale = float(bkg_y - y_pos) / obj_h
                                        try:        
                                            new_obj = obj.resize((int(obj_w*min(x_scale, y_scale)), int(obj_h*min(x_scale, y_scale))))
                                        except Exception as e:
                                            print(e)
                                        if args.annotate:
                                                # Add obj annotations
                                                annot = {
                                                                'coordinates': {
                                                                        'height': obj_h,
                                                                        'width': obj_w,
                                                                        'x': int(x_pos+(0.5*obj_w)),
                                                                        'y': int(y_pos+(0.5*obj_h))
                                                                },
                                                                'label': obj_images[i].split(".png")[0]
                                                        }
                                                obj_xmin = x_pos #xmin
                                                obj_ymin = y_pos #ymin
                                                obj_xmax = x_pos + obj_w #xmax
                                                obj_ymax = y_pos + obj_h #ymax
                                                label_fname = os.path.splitext(obj_images[i].split(".png")[0])[0]


                                                idx = label_fname.rfind("_")
                                                if idx >= 0:
                                                        final_label = label_fname[:idx]

                                                ann.append(annot)
                                                xml_data = add_xml(xml_data, obj_xmin, obj_ymin, obj_xmax, obj_ymax, final_label)


                                                # Paste the obj to the background
                                                # print("type: " + str(type(new_obj)))
                                                bkg_w_obj.paste(255, new_obj.getbbox())
                                                bkg_w_obj.paste(new_obj, (x_pos, y_pos))

                                # output_fp = output_images + str(n) + ".png"
                                # print("bkg: " + bkg)
                                # output_fp = output_images + jpg_image + "_" + str(j) + ".png"
                                output_img = output_images + bkg + "_" + str(j) + ".png"
                                output_xml = output_images + bkg + "_" + str(j) + ".xml"
                                # Save xml
                                xml_data.write(output_xml)
                                # Save image
                                # bkg_w_obj.save(fp=output_fp, format="png")
                                bkg_w_obj.save(fp=output_img, format="png")
                                if args.annotate:
                                        # Save annotation data
                                        annotations.append({
                                                "path": output_img,
                                                "annotations": ann
                                        })
                                n += 1
                            except Exception as e:
                                print(e) 
                    except Exception as e:
                        print(e)  
            except Exception as e:
                print(e)                        
        if args.annotate:
                print("Saving out Annotations", flush=True)
                # Save annotations
                with open("annotations.json", "w") as f:
                        f.write(json.dumps(annotations))

        if args.sframe:
                print("Saving out SFrame", flush=True)
                # Write out data to an sframe for turicreate training
                import turicreate as tc
                # Load images and annotations to sframes
                images = tc.load_images(output_images).sort("path")
                annots = tc.SArray(annotations).unpack(column_name_prefix=None).sort("path")
                # Join
                images = images.join(annots, how='left', on='path')
                # Save out sframe
                images[['image', 'path', 'annotations']].save("training_data.sframe")

        total_images = len([f for f in os.listdir(output_images) if not f.startswith(".")])
        print("Done! Created {} synthetic training images.".format(total_images), flush=True)
    except Exception as e:
        print(e)    