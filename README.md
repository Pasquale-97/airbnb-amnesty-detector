# Airbnb Amnesty Detector

## Summary



https://user-images.githubusercontent.com/66837999/126921956-84f899f2-4c88-4f34-a5cd-1aaea5a6ba95.mp4




- An overview:
    - A replication of the object detection system used by Airbnb to detect objects in a room. This was created by using YOLO V5 on the Open Images dataset. Talk about how many different objects and the accuracy.
    - Developed a frontend using Streamlit.
    - Wrapped in a Dockerfile and deployed to Google Cloud.

## Introduction
Artificial Intelligence is slowly integrating into companies everyday, with many of the leading tech companies leading the way. Airbnb uses deep learning on their billions of images to identify what objects are within a room as we speak. This project was inspired by Dan Bourkes 42 day project that you can [see here](https://github.com/mrdbourke/airbnb-amenity-detection). To see how Airbnb themselves went about completing the project [click here](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e). 

In this project we will create an end to end replica of Airbnb's Amnesty Detector to detect objects within a room. Python will be used for data formatting and file manipulation, YOLOv5 will be implemented for object detection, Streamlit will be used for the frontend. The application will then be wrapped in a Docker file and deployed to Google Cloud.
 

## Goals

- Achieve at least 50% mAP similar to Airbnb MVP target.
- Use Yolov5 on the same categories as Airbnb.
- Make the application user friendly by using a Streamlit frontend.
- Deploy the model to Google Cloud.
- Additional: the model can run predictions on videos as well as pictures.

## Results

- Downloaded around 30,000 images in 30 categories from Open Images and trained a YOLOv5 model in Pytorch on the data.
- Achieved a mAP of 57%, reaching the goal that we set out for.
- Created a frontend with Streamlit and deployed the model to Google Cloud.
- Model can also run predictions on videos!

Upon reflection of the project we set some goals and we achieved them! If you'd like to see how we went about replicating Airbnb's Amnesty Detector feel free to continue reading.

## Data Gathering

### Categories
![Untitled](https://user-images.githubusercontent.com/66837999/126919959-7ee69f9a-8478-4525-8f16-35c5d09c6518.png)
Mean average precision of the 30 categories used by Airbnb.

- The data will be scrapped from Openimages v6 using two different Python scripts.
- Categories consist of:
    - Toilet, "Swimming pool", Bed, "Billiard table", Sink, Fountain, Oven, "Ceiling fan", Television, "Microwave oven", "Gas stove", Refrigerator, "Kitchen & dining room table", "Washing machine", Bathtub, Stairs, Fireplace, Pillow, Mirror, Shower, Couch, Countertop, Coffeemaker, Dishwasher, "Sofa bed", "Tree house", Towel, Porch, "Wine rack", Jacuzzi

## What is Object Detection?

Object detection refers to identifying the presence of relevant objects and classifying those objects into classes. For this example we would identify the objects within a room and try to classify these objects into the relevant classes from the 30 categories defined above. Airbnb uses this method as it contains millions of images and using object detection hugely aids them in identifying what each room contains. A kitchen would potentially include items such as a microwave, dishwasher, refrigerator etc. and by using object detection we can identify these objects within the room. 

To do this we must train an object detection system to be able to recognise the relevant objects required. In this project we utilise YOLOv5 where we will try to match Airbnb's MVP. In order to train a model we must provide training data like in all deep learning projects and how we will go about this is explained below. Once the model has been trained it will be able to produce bounding boxes around the areas of interest partnered with the models confidence in the prediction made. 

## What is YOLOv5?

YOLO is an abbreviation for "You Only Look Once" and is a popular method for object detection used in the world of Computer Vision. This algorithm is popular due to its ability of detecting objects in real time. To read a full description of how YOLO works, [see here](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/), but I will provide a simple overview below:

YOLO is a combination of three techniques:

- Residual blocks
- Bounding box regression
- Intersection over union

**Residual Blocks**

Residual blocks refers to dividing the image into grids.

**Bounding box regression**

The bounding boxes highlights the areas of interest by drawing an outline around these areas. Each box consists of these features:

- Class → in this scenario bed, toilet, sink etc.
- Height
- Width
- Bounding box centre.

**Intersection over union (IOU)**

IOU is used in object detection to describe how the different boxes overlap. YOLO implements this method in order to provide the perfect ouput box for the areas of interest. For each grid, the bounding box and their confidence scores are predicted. If the IOU is equal to 1 then that means that the predicted box is equal to the real box. 

By combing these three techniques YOLO is able to provide accurate predictions in real time. YOLOv5 is the fifth implementation of the YOLO algorithm, but it comes with its controversy which we are going to bypass here. Feel free to go check it out yourself if you're curious. This implementation of YOLO is completed in Pytorch, steering away from previous models which used Darknet.

## Implementing YOLO for Object Detection

To begin making predictions we need to train a YOLO model, this is achieved by doing the following steps:

- Download a dataset → in this case we will be scrapping images from open images.
- Convert the labels of the dataset into the correct format → for this project we will be using Roboflow for a small model and Python for the full model.
- Train our YOLOv5 model in Pytorch.

### Downloading the dataset

Downloading images of the 30 categories from Open Images was obtained using the following resources:

OIDv4 → Used for 29 of the 30 categories.

OIDv6 → Used for the remaining category (kitchen and dining room table) as there were issues when using OIDv4.

**Example usage of OIDv4**

```python
!python3 ../OIDv4_ToolKit/main.py downloader --classes Billiard table --type_csv all
```

**Example usage of OIDv6**

```python
!oidv6 downloader en --dataset ../airbnb/Data --classes "Kitchen & dining room table" --type_data all
```

Note: additional installations may be required so follow the links to the resources below to download any additional requirements.

**Links to resources:**

OIDv4 → [https://github.com/EscVM/OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)

OIDv6 → [https://www.kaggle.com/getting-started/157163](https://www.kaggle.com/getting-started/157163)

### Data Formatting

![Untitled](https://user-images.githubusercontent.com/66837999/126920036-9f2e302c-eea5-4383-bd7d-80a578a36722.png)
The label file corresponding to the above image contains 2 persons (class 0) and a tie (class 27):

![Untitled-1](https://user-images.githubusercontent.com/66837999/126920040-5510995d-684b-4b3d-972a-c53d49facb70.png)
Format required to train YOLO model.

Once the images and labels have been obtained, in order to begin training a YOLO model we need to convert our labels into YOLO format (see image above). There are a variety of methods to convert the data into this required format. Resources such as Roboflow can convert images into the format required however this does cost money. In this scenario we will be using Python in order to convert all the labels in order to begin training our model. So lets take a look at a label in Open Image format.

```markdown
Jacuzzi 0.182272 560.65152 1023.7952 767.790336
```

**Steps to getting Open Image labels into YOLO format:**

- Remove spaces between words e.g. Billiard table → Billiard_table.
- Convert string of label into number format e.g. Billiard_table → 0, Jacuzzi → 1 etc.
- Convert width, height, X and Y labels into the range of 0 and 1.
- Move files into the correct format.

## Defining YOLOv5 architecture

Now that we have the labels in the correct format we can begin to train a YOLO model. YOLO provides a host of models that can be used such as:

- **YOLOv5s**
- **YOLOv5m**
- **YOLOv5l**
- **YOLOv5x**

In this scenario we decided to use the YOLOv5s due to time restrictions and efficiency as Google Colab only allows for 24 hours of training time which we have to consider once we begin training on all the data obtained. We also devised from our experiments that other models don't provide as much of an accuracy boost in compared to how much longer they take to train in this scenario. The architecture of the small model is defined below:

```markdown
# parameters
nc: 30  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, C3, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

### YOLOv5

To begin training we run the training command with the following options:

- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs.
- **data:** set the path to the required yaml file
- **cfg:** specify model configuration
- **weights:** specify a custom path to weights.
- **name:** result names
- **nosave:** only save the final checkpoint
- **cache:** cache images for faster training

### Yaml File

The yaml file is responsible for pointing the model to where the data is so that the model can be trained to detect the objects required. In this scenario we are using the airbnb-final.yaml file described below:

```markdown
train: ../airbnb/airbnb-final-dataset/train
val: ../airbnb/airbnb-final-dataset/validation

nc: 30
names: ['Sink', 'Fountain', 'Oven', 'Television', 'Refrigerator', 'Bathtub', 'Stairs', 'Fireplace', 'Pillow', 'Mirror', 'Shower', 'Couch', 'Countertop', 'Coffeemaker',
 'Dishwasher', 'Towel', 'Porch', 'Toilet', 'Jacuzzi', 'Billiard table', 'Bed', 'Ceiling fan', 'Gas stove', 'Microwave oven', 'Sofa bed', 'Swimming pool',
 'Tree house', 'Washing machine', 'Wine rack', 'kitchen_&_dining_room_table']
```

### Training a small model

In order to see whether it made sense to progress further with the project, a small model was created to see how accurate using YOLO would be on a smaller dataset. In this scenario, one category was chosen from the selection of categories in order to train a small model. In this scenario the "Billiard table" category was selected as the smaller dataset to test the model out. 

The billiard table category contained 466 ****of training images, 133 ****of validation images and 67 of test images. Once the data was obtained, and the labels had been converted into YOLOv5 format using Roboflow. The following command was run using the options above to train a YOLOv5 model:

```python
 # small model
!python3 train.py --img 640 --batch 42 --epochs 10 --data ./data/airbnb-final.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolo5x_small
```

Note: For obtaining the data for the small model Roboflow was used to get the data in the correct format, whereas Python was used when it came to training the full model.

**Results on one category (Billiard table):**

When evaluating our model, the most important metric to pay attention to is the mAP@.5. This helps us determine how accurate how model is. To read more about why this metric is important, see the link in the references at the end of this write up.

The results showed a mAP of 83% which is a promising sign for a single class. Due to this we determine that YOLOv5 is sufficient to be used in this project. Here is an example of some predictions of this small model:

![media_images_Validation_99_4](https://user-images.githubusercontent.com/66837999/126920092-448f675e-210f-4da0-bf14-fed1ba30e7f8.png)
Labelled Image → Correctly labelled image from Open Images.

![media_images_Validation_99_5](https://user-images.githubusercontent.com/66837999/126920299-29b04aa7-83de-4141-b4e0-6432b821f820.png)
Model predictions → What the model has detected within the image.

### Scaling up the model

Now that we can see from training a small model shows that there is potential in using YOLO for the project. Now we proceed with scaling up the model by downloading all 30 categories as defined above. This is achieved by running the following line of code and putting our feet up while we wait for the images to come rolling in:

```python
!python3 ../OIDv4_ToolKit/main.py downloader --classes Toilet Swimming_pool Sink Fountain Oven Ceiling_fan Television Microwave_oven Gas_stove Refrigerator Washing_machine Bathtub Stairs Fireplace Pillow Mirror Shower Couch Countertop Coffeemaker Dishwasher Sofa_bed Tree_house Towel Porch Wine_rack Jacuzzi --type_csv all
```

For some reason there were problems downloading the kitchen and dining room table category, so therefore we decided to use a different method to download this category. The following command was used:

```python
!oidv6 downloader en --dataset /content/drive/MyDrive/airbnb/Data --type_data all --classes "Kitchen & dining room table"
```

Now we run the following Python scripts:

```python
!python3 remove_space.py # remove spaces from label e.g. Billiard table
!python3 convert_annotations.py # convert file into yolo format
!python3 move_file.py # move files so it's in the correct format
```

After completing the download we now have 34,458 training images, 4538 validation images, and 751 test images.

### Training the full model

Now that we have all the data we need and converted it into the correct format we are ready to begin training our model. Using the training commands seen in the code snippet below, we begin to train our final model for 50 epochs.  

```python
# small model
!python3 train.py --img 640 --batch 42 --epochs 50 --data ./data/airbnb-final.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolo5x_small
```

## Evaluating YOLOv5 with Weights & bias

After leaving the model for around 16 hours to train, we now have a fully trained model where we can evaluate how effective it is by using Weights & Bias. The value we're particularly interested in is the mAP@0.5 to determine how well our model has done. After looking at the results we can see that the mAP reaches 0.557 or 55.7%, which is really cool! Especially considering Airbnb managed to get around 68% with extra data that we did not have access to so we're not too far off. See below for the full results of training a YOLOv5s model on around 30,000 images in 30 categories.

![media_images_Results_50_0](https://user-images.githubusercontent.com/66837999/126920345-1e995661-6216-489c-88f6-a817f60110af.png)
Evaluation of model over time using Weights & Bias

![media_images_Results_50_1](https://user-images.githubusercontent.com/66837999/126920356-7e71fae6-a4be-4c10-8543-621f51131b6b.png)
Confusion Matrix

The confusion matrix shows how accurate the model is for individual classes based on what the actual label in the image is (x-axis) and what the model predicted (y-axis). The darker the box, the more accurate the model is for that particular class.

***Results for each class:***

| Class                       | mAP @.5 |
|-----------------------------|---------|
| Refrigerator                | 0.952   |
| Swimming pool               | 0.93    |
| Ceiling fan                 | 0.909   |
| Billiard table              | 0.9     |
| Television                  | 0.834   |
| Mirror                      | 0.822   |
| Bed                         | 0.799   |
| Toilet                      | 0.769   |
| Coffeemaker                 | 0.736   |
| Washing machine             | 0.724   |
| Microwave oven              | 0.689   |
| Sofa bed                    | 0.67    |
| Sink                        | 0.628   |
| Fountain                    | 0.574   |
| Tree house                  | 0.57    |
| Fireplace                   | 0.548   |
| Stairs                      | 0.531   |
| Wine rack                   | 0.505   |
| Bathtub                     | 0.442   |
| Jacuzzi                     | 0.352   |
| Couch                       | 0.347   |
| Gas stove                   | 0.347   |
| Towel                       | 0.307   |
| Kitchen & dining room table | 0.301   |
| Countertop                  | 0.27    |
| Oven                        | 0.24    |
| Pillow                      | 0.239   |
| Porch                       | 0.177   |
| Shower                      | 0.0426  |


**Example Predictions:**

By using Weights and Bias we can look at some of the predictions the model made on a test batch. For each test batch the first image corresponds to the correctly labelled images, while the second image is the models prediction on what is in the image.

Looking at some of the predictions shows some interesting results. For example in test batch 3, the model appears to pick up an extra billiard table in the background that was missed in the labelled data (see far right image, second row in test batch 3). Also in test batch 3 it appears to have picked up some extra tv screens in the background (see bottom right image). Although the model did miss out some objects within an image, it's really interesting seeing the model pick up different objects not described in the labelled data. 

**Test batch 1:**

![media_images_Validation_49_0](https://user-images.githubusercontent.com/66837999/126920373-1c086a84-63e0-47ef-897b-d3b128773169.jpg)
test_batch0_labels.jpg - Labelled images


![media_images_Validation_49_1](https://user-images.githubusercontent.com/66837999/126920384-c2905daf-6e64-44b0-837d-a82dc381a0dc.jpg)
test_batch0_pred.jpg - Model prediction


**Test batch 2:**

![media_images_Validation_49_2](https://user-images.githubusercontent.com/66837999/126920391-e719a997-f3c6-4ba9-a8ae-e72929aa6141.jpg)
test_batch1_labels.jpg - Labelled images


![media_images_Validation_49_3](https://user-images.githubusercontent.com/66837999/126920397-f176880c-5a6a-4f9f-a015-b307cc454b0d.jpg)
test_batch1_pred.jpg - Model prediction


**Test batch 3**

![media_images_Validation_49_4](https://user-images.githubusercontent.com/66837999/126920408-1c146c63-0569-4bcf-a794-0059e46728e5.jpg)
test_batch2_labels.jpg - Labelled images


![media_images_Validation_49_5](https://user-images.githubusercontent.com/66837999/126920432-c5d28ee4-1756-46f3-bacc-45f8f480923a.jpg)
test_batch2_pred.jpg - Model prediction

## Creation of Web Application

**Exporting weights**

In order to build out a frontend for our application we need to export our weights so that we can use the model to make predictions. Luckily YOLO keeps track of all our runs and saves our best performing weights. Just what we need in this scenario. To do this we can follow one of these methods:

Method 1:

Manually click through the folders and double click on our best performing.

```markup
yolov5 → runs → your best performing model → weights → best.pt
```

Method 2:

Copy our best performing model into our drive

```python
%cp /content/yolov5/weights/best_performing_model.pt /content/drive/airbnb
```

**Streamlit**

Streamlit allows for us to bring our models to life by allowing for simple creation of a frontend of our model. Creating a frontend for our models is really important, as models tend to get left in the Jupyter notebook which doesn't allow for average person interaction. This is important to note as the majority of people using our models, whatever they may be won't know how to use Jupyter notebook. Plus, who doesn't like a cool frontend to look at. There's alternatives such as Flask [see here](https://github.com/Pasquale-97/spotify-recommender), however, in this scenario we will be using Streamlit. Before reading on, please checkout this [Github repo](https://github.com/hassan-baydoun/python_final_project) that really helped implement the front end.

Private functions:

```python
@contextmanager
def st_redirect(src, dst):
    '''
        Redirects the print of a function to the streamlit UI.
    '''
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    '''
        Sub-implementation to redirect for code redability.
    '''
    with st_redirect(sys.stdout, dst):
        yield

@contextmanager
def st_stderr(dst):
    '''
        Sub-implementation to redirect for code redability in case of errors.
    '''
    with st_redirect(sys.stderr, dst):
        yield

def _all_subdirs_of(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(bd)
    return result

def _get_latest_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(_all_subdirs_of(os.path.join('runs', 'detect')), key=os.path.getmtime)

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='airbnb-final.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='data\images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
opt = parser.parse_args()

CHOICES = {0: "Image Upload", 1: "Upload Video"}

def _save_uploadedfile(uploadedfile): 
    '''
        Saves uploaded videos to disk.
    '''
    with open(os.path.join("data", "videos",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())

def _format_func(option):
    '''
        Format function for select Key/Value implementation.
    '''
    return CHOICES[option]
```

Main function

```python
def main():
    inferenceSource = str(st.sidebar.selectbox('Select Source to detect:', options=list(CHOICES.keys()), format_func=_format_func))

    if inferenceSource == '0':
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='In progress'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)  
                picture = picture.save(f'data/images/{uploaded_file.name}') 
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='In progress'):
                st.sidebar.video(uploaded_file)
                _save_uploadedfile(uploaded_file)
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    st.title('Airbnb Amnesty Detector')
    st.header('By Pasquale Iuliano')
    inferenceButton = st.empty()

    if is_valid:
        if inferenceButton.button('Launch the Detection!'):
            with st_stdout("info"):
                detect(opt)
            if inferenceSource != '0':
                st.warning('Video playback not available on deployed version due to licensing restrictions.')
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(_get_latest_folder()):
                        st.video(f'{_get_latest_folder()}/{vid}')
                    st.balloons()
            else:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(_get_latest_folder()):
                        st.image(f'{_get_latest_folder()}/{img}')
                    st.balloons()
                    st.write("#")

    
    st.write("This application replicates [Airbnb's machine learning powered amenity detection](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) and inspired by [Dan Bourke Airbnb Amnesty Detector](https://github.com/mrdbourke/airbnb-amenity-detection).")
    st.write("To read more about how the project was completed, view on [Github](https://github.com/Pasquale-97/airbnb-amnesty-detector).")
    st.write("## How does it work?")
    st.write("Simply add an image of any room and the model will provide predictions of household items it has detected within, see example below:")
    st.image(Image.open("/Users/pasqualeiuliano/Google Drive/airbnb/yolov5/images/living-room-pred.jpeg"), 
                caption="Example of model being run on a living room.", 
                use_column_width=True)
    

if __name__ == "__main__":
    main()
```

## References

| Name                                     | Tags | Link                                                                                                       | Description                                                                       |
|------------------------------------------|------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Yolo v5 implementation                   | code | https://awesomeopensource.com/project/ultralytics/yolov5                                                   | How to implement Yolov5.                                                          |
| Streamlit                                | code | https://medium.com/analytics-vidhya/road-damage-detection-for-multiple-countries-using-yolov3-51fc7c6b43bd | Creation of Streamlit app.                                                        |
| Streamlit Example                        | code | https://github.com/hassan-baydoun/python_final_project                                                     | Template for Streamlit app.                                                       |
| Roboflow Training Yolov5 on curstom data | code | https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/                                         | Implementing Yolov5 on custom dataset.                                            |
| What is mean average precision?          | info | https://blog.roboflow.com/mean-average-precision/                                                          | Description of why mAP@.5 is the most important value to take into consideration. |
