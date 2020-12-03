import load_data
import numpy as np
import cv2


def measure_contextual_dissimilarity(c_l, c_m):
    '''
        Returns a measure of dissimilarity between two block's contextual descriptors
        Using a distance measure
        :param c_l: N_f vector
        :param c_m: N_f vector
        :return: scalar
    '''
    diff = c_l - c_m
    return np.linalg.norm(diff)

def get_context_descriptors(img, block_size, N_f=20):
    '''
        Return context descriptors for the blocks in the image. 
        context descriptors computed using multi-channel filtering.

        :param img: input image. MxMx3 => 200x200x3
        :param block_size: size of patch squares. 
            Should be perfect multiple of img dimensions. Should also be odd.
        :param N_f: (optional) number of linear filters 
        :return: context_descriptors MxMxN_f
            context_descriptors: contextual descriptors for each block in image
                loc_col, loc_row := center location of a patch/block
                context_descriptors[loc_col, loc_row]
                    := feature vector that characterizes spatial content and textures within the block
                    -> N_f dim vector
                    -> refers to the descriptor for patch img[loc_col-block_size:loc_col+block_size, loc_row-block_size:loc_row+block_size]
    '''
    d = img.shape[0]
    
    filters = np.random.rand(N_f, 3, 3) # N_f number of 3x3 filters
    con_desc = np.zeros((N_f, d, d, 3))
    blur_img = cv2.GaussianBlur(img,(5,5),0)
    
    for f_i in range(N_f):
        filt = filters[f_i]
        con_desc[f_i] = cv2.filter2D(blur_img, -1, filt) # dxd
    
    con_desc = np.moveaxis(con_desc, 0, -1) # dxdxN_f
    num_blocks = d//block_size
    half_block = block_size//2

    context_descriptors = np.zeros((num_blocks, num_blocks, 3, N_f))
    for col in range(num_blocks):
        block_col = (col * block_size) + half_block
        for row in range(num_blocks):
            block_row = (row * block_size) + half_block
            context_descriptors[col, row, :] = con_desc[block_col, block_row]
    
    return context_descriptors


def get_similar_patch_locations(patch_loc_col, patch_loc_row, context_descriptors, block_size, threshold=50.0):
    '''
        Returns a list of patch locations where the patches are contextually similar to 
        the given patch at patch_loc_col/row. 
    '''
    num_blocks = context_descriptors.shape[0]
    block_col, block_row = convert_img_center_to_block(patch_loc_col, patch_loc_row, block_size)
    c_l = context_descriptors[block_col, block_row]
    
    locations = []
    for col in range(num_blocks):
        for row in range(num_blocks):
            c_m = context_descriptors[col, row] # N_f
            diss = measure_contextual_dissimilarity(c_l, c_m)
            if diss <= threshold:
                img_col, img_row = convert_block_to_img_center(col, row, block_size)
                locations.append((img_col, img_row))
    
    return locations

def convert_block_to_img_center(block_col, block_row, block_size):
    '''
        Returns the image pixel locations from block location.
        I.e return the pixel location of the block centered at block_col,block_row

        -> convert_block_to_img_center(0, 0, 5)
        (2, 2)
        -> convert_block_to_img_center(1, 3, 5)
        (7, 17)
    '''
    half_block = block_size // 2
    col = block_col*block_size + half_block
    row = block_row*block_size + half_block
    return (col, row)


def convert_block_to_img_range(block_col, block_size):
    '''
        Returns the image pixel range from block location.
        I.e return the pixel range of of the block surrounding block_col

        The end of range is exclusive

        -> convert_block_to_img_range(0, 5)
        [0, 5)
        -> convert_block_to_img_range(3, 5)
        [15, 20)
    '''
    start = block_col*block_size
    end = start + block_size
    return (start, end)


def convert_block_center_to_img_range(block_col, block_size):
    '''
        Return the image pixel range from block center pixel location

        The end range is exclusive

        -> convert_block_center_to_img_range(2, 5)
        [0, 5)
        -> convert_block_center_to_img_range(17, 5)
        [15, 20)
    '''
    half_block = block_size // 2
    start = block_col - half_block
    end = start + block_size
    return (start, end)

def convert_img_center_to_block(img_col, img_row, block_size):
    '''
        Return the block indices of the block centered around img_col and img_row 
    
        -> convert_img_center_to_block(2, 2, 5)
        (0,0)
        -> convert_img_center_to_block(7, 17, 5)
        (1,3)
    '''
    col = img_col // block_size
    row = img_row // block_size
    return (col, row)


def combine_multi_candidate_patches(masked_img, patch_locations, block_size):
    '''
        Return a combination of candidate patches.
        Only combine source regions in patch

        :param masked_img: MxMx3 -> 200x200x3. Target regions are 0
        :param patch_locations: list[(col, row)] of center pixels of blocks
        :param block_size: size of patches
        :return: combined
            combined: avg weighted combination of all candidate patches 
    '''
    num_patches = len(patch_locations)
    combined = np.zeros((num_patches, block_size, block_size, 3))
    for i in range(num_patches):
        pot_col, pot_row = patch_locations[i]
        pot_col_start, pot_col_end = convert_block_center_to_img_range(pot_col, block_size)
        pot_row_start, pot_row_end = convert_block_center_to_img_range(pot_row, block_size)
        combined[i] = masked_img[pot_col_start:pot_col_end, pot_row_start:pot_row_end] # block_sizexblock_sizex3

    return np.mean(combined, axis=0)

def get_neighbor_patch_locations(col, row, block_size, num_blocks):
    '''
        Return locations of neighbor patches of patch at col,row

        At most, 8 neighbors (borders around a 3x3 square)

        -> get_neighbor_patch_locations(2, 2, 5)
        [(2, 7), (7, 2), (7, 7)]
    '''
    neighbors = []
    block_col, block_row = convert_img_center_to_block(col, row, block_size)

    if block_col > 0:
        new_col, new_row = convert_block_to_img_center(block_col-1, block_row, block_size)
        neighbors.append((new_col, new_row))
        if block_row > 0:
            new_col, new_row = convert_block_to_img_center(block_col-1, block_row-1, block_size)
            neighbors.append((new_col, new_row))
        if block_row < num_blocks-1:
            new_col, new_row = convert_block_to_img_center(block_col-1, block_row+1, block_size)
            neighbors.append((new_col, new_row))
    if block_row > 0:
        new_col, new_row = convert_block_to_img_center(block_col, block_row-1, block_size)
        neighbors.append((new_col, new_row))
    if block_row < num_blocks-1:
        new_col, new_row = convert_block_to_img_center(block_col, block_row+1, block_size)
        neighbors.append((new_col, new_row))
    if block_col < num_blocks-1:
        new_col, new_row = convert_block_to_img_center(block_col+1, block_row, block_size)
        neighbors.append((new_col, new_row))
        if block_row > 0:
            new_col, new_row = convert_block_to_img_center(block_col+1, block_row-1, block_size)
            neighbors.append((new_col, new_row))
        if block_row < num_blocks-1:
            new_col, new_row = convert_block_to_img_center(block_col+1, block_row+1, block_size)
            neighbors.append((new_col, new_row))

    return neighbors


def get_patches_to_fill(mask, block_size):
    '''
        Returns a list of patch locations for patches that need to be filled (i.e, at least one pixel missing)

        :param mask: MxM. mask[x,y] = True, means image @ (x,y) needs to be filled
        :param block_size: dimension of patches
        :return: list[(col, row)] where col, row are the center locations of patches
    '''
    d = mask.shape[0]
    num_blocks = d//block_size
    # Locations of blocks that have at least one pixel needing to be filled
    patch_locations = []
    for col in range(num_blocks):
        img_col_start, img_col_end = convert_block_to_img_range(col, block_size)
        for row in range(num_blocks):
            img_row_start , img_row_end = convert_block_to_img_range(row, block_size)
            mask_patch = mask[img_col_start:img_col_end, img_row_start:img_row_end]
            if np.any(mask_patch):
                block_col, block_row = convert_block_to_img_center(col, row, block_size)
                patch_locations.append((block_col, block_row))
    return patch_locations


def get_candidate_patches_for_locations(patch_locations, masked_img, mask, block_size, context_descriptors):
    '''
        Returns combined candidate patches for each patch location
        each candidate patch is block_size x block_size x 3
    '''
    num_loc = len(patch_locations)
    candidates = np.zeros((num_loc, block_size, block_size, 3))
    total_block_size = block_size**2  # How many pixels inside the block
    for loc_i in range(num_loc):
        (block_col, block_row) = patch_locations[loc_i]
        img_col_start, img_col_end = convert_block_center_to_img_range(block_col, block_size)
        img_row_start, img_row_end = convert_block_center_to_img_range(block_row, block_size)
        mask_patch = mask[img_col_start:img_col_end, img_row_start:img_row_end]
        num_unknown = np.sum(mask_patch)
        if num_unknown / total_block_size < 0.5:
            # reliable
            img_cpy[img_col_start:img_col_end, img_row_start:img_row_end] = [0, 0, 255]
            similar_patches = get_similar_patch_locations(block_col, block_row, context_descriptors, block_size)
        else:
            # Unreliable block
            # TODO: Fix: infinite recursion.. try to implement memoization. save the combined patches so we dont keep recalculating
            img_cpy[img_col_start:img_col_end, img_row_start:img_row_end] = [0, 255, 0]
            neighbor_patches = get_neighbor_patch_locations(block_col, block_row, block_size, num_blocks)
            similar_patches = get_candidate_patches_for_locations(neighbor_patches, masked_img, mask, block_size, context_descriptors)
        candidates[loc_i] = combine_multi_candidate_patches(masked_img, similar_patches, block_size)
    return candidates


if __name__ == '__main__':
    test_data = load_data.load_test_data()
    train_data = load_data.load_train_data()

    img = train_data[0] # 200x200x3
    d = img.shape[0]
    mask = np.zeros((d, d), dtype=np.bool)
    mask[79:152, 65:77] = True

    img_cpy = np.copy(img)
    img_cpy[mask.nonzero()] = 0
    
    masked_img = np.copy(img_cpy)
    final_img = np.copy(masked_img)
    
    block_size = 5
    num_blocks = d//block_size
    half_block = block_size//2
    
    # Add patch grid lines
    for col in range(1, num_blocks):
        img_col = col*block_size
        img_cpy[img_col, :] = [255, 0, 0]
        img_cpy[:, img_col] = [255, 0, 0]
    
    patch_locations = get_patches_to_fill(mask, block_size)
    
    context_descriptors = get_context_descriptors(masked_img, block_size) # num_blocks x num_blocks x N_f
    
    candidate_patches = get_candidate_patches_for_locations(patch_locations, masked_img, mask, block_size, context_descriptors)
    
    for loc_i in range(len(patch_locations)):
        (block_col, block_row) = patch_locations[loc_i]
        candidate = candidate_patches[loc_i]
        img_col_start, img_col_end = convert_block_center_to_img_range(block_col, block_size)
        img_row_start, img_row_end = convert_block_center_to_img_range(block_row, block_size)
        mask_patch = mask[img_col_start:img_col_end, img_row_start:img_row_end]
        final_img[img_col_start:img_col_end, img_row_start:img_row_end][mask_patch, :] = candidate[mask_patch, :]
        img_cpy[img_col_start:img_col_end, img_row_start:img_row_end] = [0, 0, 255]
    
    cv2.imshow("patches filled in", final_img)
    cv2.imshow("which patches were filled in", img_cpy)
    cv2.waitKey(0)
