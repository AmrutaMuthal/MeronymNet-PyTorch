import numpy as np

def arrangement(a, b, object_name):
    if object_name=='cow' or object_name=='sheep':
        p = [ 10,11,18,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='bird':
        p = [ 10,11,12,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='person':
        p = [ 10,11,19,18,20,22,21,23,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='cat':
        p = [ 10,11,13,12,14,16,15,9,0,7,3,4,5,6,1,2,8]
    elif object_name=='dog':
        p = [ 10,11,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8]
    elif object_name=='horse':
        p = [ 10,11,19,18,20,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='aeroplane':
        p = [ 10,11,19,18,20,22,21,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='car':
        p = [ 10,11,19,18,20,22,21,23,24,25,26,27,28,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='motorbike':
        p = [ 10,11,13,12,14,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='bicycle':
        p = [ 10,11,13,12,14,15,9,0,7,3,4,5,6,1,2,8 ]
    else:
      print("error")
    return a[p], b[p]


def rearrange(lbl, bbx, mask, object_name):
    if object_name=='cow' or object_name=='sheep':
        p = np.asarray([1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19])-1
    elif object_name=='bird':
        p = np.asarray([1,3,2,4,5,6,8,7,11,12,9,10,13])-1
    elif object_name=='person':
        p = np.asarray([1,3,2,5,4,7,6,8,9,10,11,12,16,17,18,13,14,15,22,23,24,19,20,21])-1
    elif object_name=='cat':
        p = np.asarray([1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17])-1
    elif object_name=='dog':
        p = np.asarray([1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17,18])-1
    elif object_name=='horse':
        p = np.asarray([1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19,21,20])-1
    elif object_name=='aeroplane':
        p = np.asarray([1,3,2,5,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])-1
    elif object_name=='car':
        p = np.asarray([1,3,2,4,5,7,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])-1
    elif object_name=='motorbike':
        p = np.asarray([1,3,2,4,5,7,6,8,9,10,11,12,13,14,15])-1
    elif object_name=='bicycle':
        p = np.asarray([1,3,2,4,5,7,6,8,9,10,11,12,13,14,15,16])-1
    else:
      print("error")    
    return lbl[p], bbx[p], mask[p]


def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)
    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def bounder(img):
    result = np.where(img<0.5)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        img[cord] = 0
    result1 = np.where(img>=0.5)
    listOfCoordinates1 = list(zip(result1[0], result1[1]))
    for cord in listOfCoordinates1:
        img[cord] = 1
    return img

def add_images(canvas,img, ii):
    result = np.where(img!=0)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        canvas[cord] = ii
    return canvas



def label_2_image(img):
    rgb_img = np.zeros((img.shape[0],img.shape[1], 3)) 
    for key in label_to_color.keys():
        rgb_img[img == key] = label_to_color[key]
    return rgb_img

def make_mask(box,mask):    
    b_in = np.copy(box)
    mx = np.copy(mask)
    max_parts = len(box)
    xmax = max(box[:,2])
    ymax = max(box[:,3])
    canvas = np.zeros((int(ymax),  int(xmax)), np.float32)
    b_in, mx = arrangement(b_in, mx,object_name)
    for i in range(max_parts): 
        x_min, y_min, x_max, y_max = b_in[i]
        if x_max-x_min > 0 and y_max-y_min>0:
            x, y = canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ].shape
            canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ] = add_images(canvas[ int(y_min):int(y_max), int(x_min):int(x_max)  ],cv2.resize(bounder(np.squeeze(mx[i]))*(i+1), (y,x)), i+1)
    plt.imshow(label_2_image(canvas))
    plt.show()
    return label_2_image(canvas)
    
def plot_image_bbx(bbx,image):
    canvas = np.copy(image)
    i = 0
    for coord in bbx:
        x_minp, y_minp,x_maxp , y_maxp= coord
        if [x_minp, y_minp,x_maxp , y_maxp]!=[0,0,0,0]:
            cv2.rectangle(canvas, ((x_minp), (y_minp)), ((x_maxp) , (y_maxp) ), colors[i], 4)
        i = i+1
    plt.imshow(canvas)
    plt.show()
    return canvas

def flip_mask(mask):
    mx = np.fliplr(mask)
    return mx

def flip_bbx(label, bbx):
    bx = np.copy(bbx)
    x_min = min(bbx[:,0])
    y_min = min(bbx[:,1])
    x_max = max(bbx[:,2])
    y_max = max(bbx[:,3])
    img_center = np.asarray( [((x_max+x_min)/2),  ((y_max+y_min)/2)] )
    img_center = np.hstack( (img_center, img_center) )
    bx[:,[0,2]] += 2*(img_center[[0,2]] - bx[:,[0,2]])
    box_w = abs(bx[:,0] - bx[:,2])
    bx[:,0] -= box_w
    bx[:,2] += box_w
    for i in range(len(label)):
        if sum(label[i])==0:
            bx[i][0] = 0
            bx[i][1] = 0
            bx[i][2] = 0
            bx[i][3] = 0
    return bx

def flip_data_instance(label, box, mask):
    bx = np.copy(flip_bbx(label,box))
    mx = np.fliplr(mask)
    #ix = np.copy(image[:,::-1])
    lx = np.copy(label)
    lx, bx, mx = rearrange(lx, bx, mx,object_name)
    return lx,bx,mx
def cordinates(img):    
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0

    for i in img:
        if np.count_nonzero(i)!=0:
            break
        y_min+=1
        
    for i in img.T:
        if np.count_nonzero(i)!=0:
            break
        x_min+=1
    
    for i in img[::-1]:
        if np.count_nonzero(i)!= 0:
            break
        y_max+=1
    y_max = img.shape[0] - y_max - 1
    
    for i in img.T[::-1]:
        if np.count_nonzero(i) != 0:
            break
        x_max+=1
    x_max = img.shape[1] - x_max - 1

    return x_min, y_min, x_max, y_max

def rotate_im(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image

def get_corners(bboxes):
    
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]


    return bbox

def rotate_box(corners,angle,  cx, cy, h, w):

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated

def get_enclosing_box(corners):
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def rtt(angle, label, img, bboxes):

    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    img = rotate_im(img, angle)

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:,4:]))


    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)


    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w,h))
    
    new_bbox[:,:4] = np.true_divide(new_bbox[:,:4], [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]) 

    for i in range(len(label)):
        if sum(label[i])==0:
            new_bbox[i][0] = 0
            new_bbox[i][1] = 0
            new_bbox[i][2] = 0
            new_bbox[i][3] = 0
    
    return img, new_bbox
def render_mask(box,mask,angle):
    mx = np.copy(mask)
    b_in = np.copy(box)
    max_parts = len(box)
    xmax = max(box[:,2])
    ymax = max(box[:,3])
    temp_mx_list = []
    temp_bx_list = []
    for i in range(max_parts):
        canvas = np.zeros((int(ymax),  int(xmax)), np.float32)
        x_min, y_min, x_max, y_max = b_in[i]
        if x_max-x_min > 0 and y_max-y_min>0:
            
            x, y = canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ].shape
            canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ] = add_images(canvas[ int(y_min):int(y_max), int(x_min):int(x_max)  ],cv2.resize(bounder(np.squeeze(mx[i]))*(i+1), (y,x)), i+1)
            canvas = rotate_im(canvas,angle)
            x_min, y_min, x_max, y_max = cordinates(canvas)
            #canvas = canvas[int(y_min):int(y_max), int(x_min):int(x_max)]
            #resized_cropped = np.expand_dims(cv2.resize(canvas, (64, 64)), axis = 3)
        temp_bx_list.append([x_min, y_min, x_max, y_max])
        #temp_mx_list.append(resized_cropped)
        #plt.imshow(canvas)
        #plt.show()
    return np.asarray(temp_bx_list,dtype="float32")

def centre_object(bbx,canvas_size):
    
    pos = get_pos(bbx)
    bx = np.copy(bbx)
    
    h,w = canvas_size
    
    h_o = max(bbx[pos[:, 0]==1, 3]) + min(bbx[pos[:, 0]==1, 1])
    w_o = max(bbx[pos[:, 0]==1, 2]) + min(bbx[pos[:, 0]==1, 0])
    
    h_shift = int(h/2 - h_o/2)
    w_shift = int(w/2 - w_o/2)
    
    bx[:,0] = (bx[:,0]+w_shift)
    bx[:,1] = (bx[:,1]+h_shift)
    bx[:,2] = (bx[:,2]+w_shift)
    bx[:,3] = (bx[:,3]+h_shift)

    return bx*pos

def centre_object_old(bbx,canvas_size):
    
    pos = get_pos(bbx)
    bx = np.copy(bbx)
    
    h,w = canvas_size
    
    h_o = max(bbx[:, 3]) + min(bbx[:, 1])
    w_o = max(bbx[:, 2]) + min(bbx[:, 0])
    
    h_shift = int(h/2 - h_o/2)
    w_shift = int(w/2 - w_o/2)
    
    bx[:,0] = (bx[:,0]+w_shift)
    bx[:,1] = (bx[:,1]+h_shift)
    bx[:,2] = (bx[:,2]+w_shift)
    bx[:,3] = (bx[:,3]+h_shift)

    return bx*pos

def scale(bbx, scaling_factor):
    
    height = max(bbx[:,3])
    width = max(bbx[:,2])
    
    pos = get_pos(bbx)
    
    fold_a = np.copy(bbx)
    fold_b = np.copy(bbx)
    fold_c = np.copy(bbx)
    fold_d = np.copy(bbx)
    
    center_shift_pos = canvas_size*(scaling_factor)
    center_shift_neg = canvas_size*(-scaling_factor)
    
    fold_a = fold_a*(1-scaling_factor) + center_shift_pos
    fold_a = centre_object(fold_a*pos, (canvas_size, canvas_size))
    fold_b = fold_b*(1+scaling_factor) + center_shift_neg
    fold_b = centre_object(fold_b*pos, (canvas_size, canvas_size))
    
    
    center_shift_pos = canvas_size*(scaling_factor*0.5)
    center_shift_neg = canvas_size*(-scaling_factor*0.5)
    
    fold_c = fold_c*(1-scaling_factor*0.5) + center_shift_pos
    fold_c = centre_object(fold_c*pos, (canvas_size, canvas_size))
    fold_d = fold_d*(1+scaling_factor*0.5) + center_shift_neg
    fold_d = centre_object(fold_d*pos, (canvas_size, canvas_size))
     
    return fold_a*pos,fold_b*pos,fold_c*pos,fold_d*pos

def scale_old(bbx, scaling_factor):
    
    height = max(bbx[:,3])
    width = max(bbx[:,2])
    
    pos = get_pos(bbx)
    
    fold_a = np.copy(bbx)
    fold_b = np.copy(bbx)
    fold_c = np.copy(bbx)
    fold_d = np.copy(bbx)
    
    scale_height = scaling_factor
    scale_width = scaling_factor
    
    fold_a[:,0] = (fold_a[:,0]-scale_width)
    fold_b[:,1] = (fold_b[:,1]-scale_height)
    fold_c[:,2] = (fold_c[:,2]+scale_width)
    fold_d[:,3] = (fold_d[:,3]+scale_height)
    
    return fold_a*pos,fold_b*pos,fold_c*pos,fold_d*pos

def append_labels(box):
  all_box = []
  for bbx in box:
    pos = get_pos(bbx)
    bbx = (((bbx/canvas_size)))*pos
    #calculate (x_c,y_c, w,h)
    #eps = 0.00001
    #bbx[:,2] = bbx[:,2] + eps
    #bbx[:,3] = bbx[:,3] + eps
    #bxx = np.copy(bbx)
    ##bxx[:,0] = (bbx[:,2] + bbx[:,0])/2
    ##bxx[:,1] = (bbx[:,3] + bbx[:,1])/2
    #bxx[:,0] =  np.log(bbx[:,0])
    #bxx[:,1] =  np.log(bbx[:,1])
    #bxx[:,2] = (np.log(abs(bbx[:,3] - bbx[:,1])))
    #bxx[:,3] = (np.log(abs(bbx[:,2] - bbx[:,0])))
    #bbx = bxx*pos
    #bbx[np.isnan(bbx)] = 0

    temp = []
    for bx in bbx:
      if bx.tolist()!=[0,0,0,0]:
        temp.append([1]+bx.tolist())
      else:
        temp.append([0]+bx.tolist())
    all_box.append(temp)
  return np.asarray(all_box)

def plot_bbx(bbx):
    bbx = bbx*canvas_size
    canvas = np.ones((canvas_size,canvas_size,3), np.uint8) * 255
    for i, coord in enumerate(bbx):
        x_minp, y_minp,x_maxp , y_maxp= coord
        if [x_minp, y_minp,x_maxp , y_maxp]!=[0,0,0,0]:
            cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp) , int(y_maxp) ), colors[i], 6)
    return canvas

def transform_bbx(bbx1):
    
    eps = 0.00001
    bbx = np.copy(bbx1)
    bxx = np.copy(bbx)

    bbx[:,0] = np.exp(bbx[:,0])
    bbx[:,1] = np.exp(bbx[:,1])
    bbx[:,2] = np.exp(bbx[:,2])
    bbx[:,3] = np.exp(bbx[:,3])
    
    bxx[:,0] = bbx[:,0]
    bxx[:,1] = bbx[:,1]
    bxx[:,2] = bbx[:,0] + (bbx[:,3]) 
    bxx[:,3] = bbx[:,1] + (bbx[:,2]) 
    
    return bxx