# 3D-Brain-Tumor-Auto-Segmentation

* In this repo, I created 3D Segmentation Model to identify Brain Tumor.

* Brain MRIs are in 3D volumes so it is very difficult to pass the whole 3D volume at a time. To overcome this issue I divided 3d volumes into 3D sub volumes and selected only sub volumes which have at least 5% Tumor area.

* To create Tensor dataset in I created a notebook in which MRI images which are in **.nii.gz** forms is converted and pre-processed(sub-volumes and Standardization) and then saved for training.

* [Pre-trained Model](https://github.com/vkmavani/3D-Brain-Tumor-Auto-Segmentation/blob/main/TumorSegmentation.pt)

### **NOTE**
  
  * Due to the lack of Computional Resources, I didn't use whole dataset.
