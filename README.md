# Automatic Segmentation of Calcified Plaque in Carotid Arteries

We introduce a two-step segmentation process. First, segments the carotid artery to narrow the search space and focus on the region of interest around the artery. Then, it segments the calcified plaque within that targeted region. This approach achieves an intersection over union (IoU) of 0.9412 for the 2D model and 0.8095 for the 3D model, outperforming the baseline methods that directly segment plaques.

<img width="874" alt="Screenshot 2024-10-31 at 12 48 46 AM" src="https://github.com/user-attachments/assets/39c69130-bc3f-4fcb-a127-ec817c2217c0">

## Table of Contents
* [Automatic Segmentation of Calcified Plaque in Carotid Arteries](#cactas-project)
  * [Table of Contents](#table-of-contents)
  * [Carotid Artery Segmentation](#ca-seg)
  * [Plaque Segmentation](#plaque-seg)
* [License](#license)
* [Contact](#contact)

### Carotid Artery Segmentation
First, using TotalSegmentator, we get the output of the common carotid artery left and right, vertebrae C3 and C5. Then we analyzed carotid artery intensity values and calculated a threshold based on the mean intensity. Then we set threshold ranges of -50 and +200. The images were then cropped around the carotid artery, guided by vertebrae segmentation from TotalSegmentator. Finally, we applied a region-growing technique to produce the carotid artery mask.

You can find the code here - [Carotid Artery Segmentation](https://github.com/jiehyunjkim/CACTAS-AI/tree/main/_EXPERIMENTS/CarotidArterySegmentation)

<img width="908" alt="Screenshot 2024-10-31 at 12 52 34 AM" src="https://github.com/user-attachments/assets/70a1b73a-7232-4f7d-96c1-56f12934a65a">

### Plaque Segmentation
We segment plaque using 2D and 3D UNet. You can find the code here - [2D UNet](https://github.com/jiehyunjkim/CACTAS-AI/tree/main/_EXPERIMENTS/2D%20UNet), [3D UNet](https://github.com/jiehyunjkim/CACTAS-AI/tree/main/_EXPERIMENTS/3D%20UNet)

2D UNet:<br/>
<img width="920" alt="Screenshot 2024-10-31 at 12 52 44 AM" src="https://github.com/user-attachments/assets/0ee408ef-b3f0-4275-b6ad-12732351594d"><br/>

3D UNet:<br/>
<img width="908" alt="Screenshot 2024-10-31 at 12 52 50 AM" src="https://github.com/user-attachments/assets/6972d42e-1653-4d65-a383-3187ddaa7a79"><br/>


## License 
This project is licensed under the MIT License - see the [LICENSE](https://github.com/jiehyunjkim/CACTAS/blob/main/LICENSE) file for details.

## Contact
For any questions or comments, feel free to reach out to:
  * Jenna Kim at JieHyun.Kim001@umb.edu
