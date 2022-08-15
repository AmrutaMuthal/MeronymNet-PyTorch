

max_num_node = 24
canvas_size = 660

flip_bird = [1,3,2,4,5,6,8,7,11,12,9,10,13]
flip_cow = [1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19]
flip_cat = [1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17]
flip_dog = [1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17,18]
flip_horse = [1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19,21,20]
flip_person = [1,3,2,5,4,7,6,8,9,10,11,12,16,17,18,13,14,15,22,23,24,19,20,21]
flip_sheep = flip_cow

person_tree = {}
person_tree[0] = [1,2,3,4,7,8,9,10,11]
person_tree[1] = [0,2,3,5,7]
person_tree[2] = [0,1,4,6,7]
person_tree[3] = [0,1]
person_tree[4] = [0,2]
person_tree[5] = [1]
person_tree[6] = [2]
person_tree[7] = [0,1,2,8]
person_tree[8] = [0,7]
person_tree[9] = [0]
person_tree[10] = [0,11,13,16,19,22]
person_tree[11] = [0,10]
person_tree[12] = [13,14]
person_tree[13] = [10,12]
person_tree[14] = [12]
person_tree[15] = [16,17]
person_tree[16] = [10,15]
person_tree[17] = [15]
person_tree[18] = [19,20]
person_tree[19] = [18,10]
person_tree[20] = [18]
person_tree[21] = [22,23]
person_tree[22] = [21,10]
person_tree[23] = [21]

bird_tree = {}
bird_tree[0] = [1,2,3,4,5]
bird_tree[1] = [0,2,3]
bird_tree[2] = [0,1,3]
bird_tree[3] = [0,1,2]
bird_tree[4] = [0,5,6,7,8,10,12]
bird_tree[5] = [0,4]
bird_tree[6] = [7,4]
bird_tree[7] = [6,4]
bird_tree[8] = [4,9]
bird_tree[9] = [8]
bird_tree[10] = [11,4]
bird_tree[11] = [10]
bird_tree[12] = [4] 

dog_tree = {}
dog_tree[0] = [1,2,3,4,5,6,7,17]
dog_tree[1] = [2,0,3]
dog_tree[2] = [0,1,4]
dog_tree[3] = [0,1]
dog_tree[4] = [0,2]
dog_tree[5] = [0,1,2]
dog_tree[6] = [0,8,10,12,14,16,7]
dog_tree[7] = [0,6]
dog_tree[8] = [9,6]
dog_tree[9] = [8]
dog_tree[10] = [6,11]
dog_tree[11] = [10]
dog_tree[12] = [13,6]
dog_tree[13] = [12]
dog_tree[14] = [6,15]
dog_tree[15] = [14]
dog_tree[16] = [6]
dog_tree[17] = [0]

cat_tree = {}
cat_tree[0] = [1,2,3,4,5,6,7]
cat_tree[1] = [2,0,3]
cat_tree[2] = [0,1,4]
cat_tree[3] = [0,1]
cat_tree[4] = [0,2]
cat_tree[5] = [0,1,2]
cat_tree[6] = [0,8,10,12,14,16,7]
cat_tree[7] = [0,6]
cat_tree[8] = [9,6]
cat_tree[9] = [8]
cat_tree[10] = [6,11]
cat_tree[11] = [10]
cat_tree[12] = [13,6]
cat_tree[13] = [12]
cat_tree[14] = [6,15]
cat_tree[15] = [14]
cat_tree[16] = [6]

horse_tree = {}
horse_tree[0] = [1,2,3,4,5,8,9]
horse_tree[1] = [2,0,3]
horse_tree[2] = [0,1,4]
horse_tree[3] = [0,1]
horse_tree[4] = [0,2]
horse_tree[5] = [0,1,2]
horse_tree[6] = [11] #lfho
horse_tree[7] = [13] #rfho
horse_tree[8] = [0,10,12,14,16,18]
horse_tree[9] = [0,8]
horse_tree[10] = [8,11,12]
horse_tree[11] = [10,6]
horse_tree[12] = [10,8,13]
horse_tree[13] = [7]
horse_tree[14] = [8,15,16]
horse_tree[15] = [14,19]
horse_tree[16]= [14,17]
horse_tree[17]= [16,20]
horse_tree[18]= [8]
horse_tree[19] = [15]
horse_tree[20] = [17]

cow_tree = {}
cow_tree[0] = [1,2,3,4,5,6,7,8,9]
cow_tree[1] = [2,0,3,5]
cow_tree[2] = [0,1,4,5]
cow_tree[3] = [0,1,6]
cow_tree[4] = [0,2,7]
cow_tree[5] = [0,1,2]
cow_tree[6] = [0,3] #lfho
cow_tree[7] = [0,4,13] #rfho
cow_tree[8] = [0,9,10,12,14,16,18]
cow_tree[9] = [0,8]
cow_tree[10] = [8,11,12]
cow_tree[11] = [10,6]
cow_tree[12] = [10,8,13]
cow_tree[13] = [7,12]
cow_tree[14] = [8,15,16]
cow_tree[15] = [14]
cow_tree[16]= [8,14,17]
cow_tree[17]= [16]
cow_tree[18]= [8]

motorbike_tree = {}
motorbike_tree[0] = [14,1,2]
motorbike_tree[1] = [14,0]
motorbike_tree[2] = [14,0]
motorbike_tree[3] = [14]
motorbike_tree[4] = [14]
motorbike_tree[5] = [14]
motorbike_tree[6] = [14]
motorbike_tree[7] = [14]
motorbike_tree[8] = [14]
motorbike_tree[9] = [14]
motorbike_tree[10]= [14]
motorbike_tree[11]= [14]
motorbike_tree[12]= [14]
motorbike_tree[13]= [14]
motorbike_tree[14]= [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

bicycle_tree = {}
bicycle_tree[0] = [15,1,3]
bicycle_tree[1] = [15,0,2,4]
bicycle_tree[2] = [15,1]
bicycle_tree[3] = [15,0]
bicycle_tree[4] = [15,1]
bicycle_tree[5] = [15]
bicycle_tree[6] = [15]
bicycle_tree[7] = [15]
bicycle_tree[8] = [15]
bicycle_tree[9] = [15]
bicycle_tree[10]= [15]
bicycle_tree[11]= [15]
bicycle_tree[12]= [15]
bicycle_tree[13]= [15]
bicycle_tree[14]= [15]
bicycle_tree[15]= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

aeroplane_tree = {}
aeroplane_tree[0] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
aeroplane_tree[1] = [0]
aeroplane_tree[2] = [0,3]
aeroplane_tree[3] = [0,2]
aeroplane_tree[4] = [0]
aeroplane_tree[5] = [0]
aeroplane_tree[6] = [0]
aeroplane_tree[7] = [0]
aeroplane_tree[8] = [0]
aeroplane_tree[9] = [0]
aeroplane_tree[10]= [0]
aeroplane_tree[11]= [0]
aeroplane_tree[12]= [0]
aeroplane_tree[13]= [0]
aeroplane_tree[14]= [0]
aeroplane_tree[15]= [0]
aeroplane_tree[16]= [0]
aeroplane_tree[17]= [0]
aeroplane_tree[18]= [0]
aeroplane_tree[19]= [0]
aeroplane_tree[20]= [0]
aeroplane_tree[21]= [0]
aeroplane_tree[22]= [0]

tree = { 'aeroplane':aeroplane_tree, 'motorbike':motorbike_tree,'bicycle':bicycle_tree, 'person':person_tree, 'cow':cow_tree, 'dog':dog_tree, 'cat':cat_tree, 'sheep':cow_tree, 'bird':bird_tree, 'horse':horse_tree }

object_names = ['cow','sheep','bird','person','cat','dog','horse','aeroplane','motorbike','bicycle']

class_dic = {'cow':0,'sheep':1,'bird':2,'person':3,'cat':4,'dog':5,'horse':6,'aeroplane':7,'motorbike':8,'bicycle':9,'car':10}

colors = [(1, 0, 0),
          (0.737, 0.561, 0.561),
          (0.255, 0.412, 0.882),
          (0.545, 0.271, 0.0745),
          (0.98, 0.502, 0.447),
          (0.98, 0.643, 0.376),
          (0.18, 0.545, 0.341),
          (0.502, 0, 0.502),
          (0.627, 0.322, 0.176),
          (0.753, 0.753, 0.753),
          (0.529, 0.808, 0.922),
          (0.416, 0.353, 0.804),
          (0.439, 0.502, 0.565),
          (0.784, 0.302, 0.565),
          (0.867, 0.627, 0.867),
          (0, 1, 0.498),
          (0.275, 0.51, 0.706),
          (0.824, 0.706, 0.549),
          (0, 0.502, 0.502),
          (0.847, 0.749, 0.847),
          (1, 0.388, 0.278),
          (0.251, 0.878, 0.816),
          (0.933, 0.51, 0.933),
          (0.961, 0.871, 0.702)]
colors = (np.asarray(colors)*255)


label_to_color = {0:(0,0,0),
                    1:(0.941, 0.973, 1),
                    2:(0.98, 0.922, 0.843),
                    3:(0, 1, 1),
                    4:(0.498, 1, 0.831),
                    5:(0.941, 1, 1),
                    6:(0.961, 0.961, 0.863),
                    7:(1, 0.894, 0.769),
                    8:(0.251, 0.878, 0.816),
                    9:(1, 0.388, 0.278),
                    10:(0, 0, 1),
                    11:(0.541, 0.169, 0.886),
                    12:(0.647, 0.165, 0.165),
                    13:(0.871, 0.722, 0.529),
                    14:(0.373, 0.62, 0.627),
                    15:(0.498, 1, 0),
                    16:(0.824, 0.412, 0.118),
                    17:(1, 0.498, 0.314),
                    18:(0.392, 0.584, 0.929),
                    19:(0.275, 0.51, 0.706),
                    20:(0.863, 0.0784, 0.235),
                    21:(0, 1, 1),
                    22:(0, 0, 0.545),
                    23:(0.824, 0.706, 0.549),
                    24:(0.251, 0.878, 0.816)}