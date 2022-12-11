from skimage.metrics import structural_similarity
import cv2

def orb_sim(img1, img2):
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def structural_sim(img1, img2):

  sim, diff = structural_similarity(img1, img2, full=True)
  return sim

compared = []
top_five_matches = []
for i in range(1,11) :
      similar_to_img1 = []
      for j in range(1,11):
        
        # To prevent from already compared images  
        if [j,i] in compared :
            similar_to_img1.append([0, [i, j]])
            continue
        compared.append([i,j])
        
        # To prevent from image compare with the same image
        if i == j :
            similar_to_img1.append([0, [i, j]])
            continue
          
        img1 = cv2.imread('Images/cat'+str(i)+'.png', 0)  
        img2 = cv2.imread('Images/cat'+str(j)+'.png', 0)
        orb_similarity =  orb_sim(img1, img2)  
        similar_to_img1.append([orb_similarity,[i,j]])

      # sort based on the orb similarity 
      if similar_to_img1 != []:
        similar_to_img1.sort(key = lambda x : x[0])
        
      # most similar images 
      most_similar = similar_to_img1[-1]
      top_five_matches.append(most_similar)
      
# top five images with most similar 
top_five_matches.sort(key = lambda x : x[0])
top_five_matches = top_five_matches[-5:]

for i in reversed(top_five_matches) :
      print(f'Similarity of Image{i[1][0]} and Image{i[1][1]} is {i[0]}')

