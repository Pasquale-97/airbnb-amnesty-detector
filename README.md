# airbnb-amnesty-detector

## Summary

- Include a screen recording of the application.
- An overview of what the project is:
    - A replication of the object detection system used by Airbnb to detect objects in a room. This was created by using YOLO V5 on the Open Images dataset. Talk about how many different objects and the accuracy.
    - Developed a frontend using Streamlit.
    - Wrapped in a Dockerfile and deployed to Google Cloud.

## Introduction

 

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

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8359eebd-c554-42fd-837e-55aba24d5625/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8359eebd-c554-42fd-837e-55aba24d5625/Untitled.png)

Mean average precision of the 30 categories used by Airbnb.

- The data will be scrapped from Openimages v6 using two different Python scripts.
- Categories consist of:
    - Toilet, "Swimming pool", Bed, "Billiard table", Sink, Fountain, Oven, "Ceiling fan", Television, "Microwave oven", "Gas stove", Refrigerator, "Kitchen & dining room table", "Washing machine", Bathtub, Stairs, Fireplace, Pillow, Mirror, Shower, Couch, Countertop, Coffeemaker, Dishwasher, "Sofa bed", "Tree house", Towel, Porch, "Wine rack", Jacuzzi

## What is Object Detection?

Object detection refers to identifying the presence of relevant objects and classifying those objects into classes. For this example we would identify the objects within a room and try to classify these objects into the relevant classes from the 30 categories defined above. Airbnb uses this method as it contains millions of images and using object detection hugely aids them in identifying what each room contains. A kitchen would potentially include items such as a microwave, dishwasher, refrigerator etc. and by using object detection we can identify these objects within the room. 

To do this we must train an object detection system to be able to recognise the relevant objects required. In this project we utilise YOLOv5 where we will try to match Airbnb's MVP. In order to train a model we must provide training data like in all deep learning projects and how we will go about this is explained below. Once the model has been trained it will be able to produce bounding boxes around the areas of interest partnered with the models confidence in the prediction made. 

## What is YOLOv5?

YOLO is an abbreviation for "You Only Look Once" and is a popular method for object detection used in the world of Computer Vision. This algorithm is popular due to its ability of detecting objects in real time. To read a full description of how YOLO works, see [here]([https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/)) but I will provide a simple overview below:

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

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a253aec2-7c8b-41ab-88e4-70feba11937c/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a253aec2-7c8b-41ab-88e4-70feba11937c/Untitled.png)

The label file corresponding to the above image contains 2 persons (class 0) and a tie (class 27):

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5cd8c33-eac2-4620-8378-01de3c9ffb5f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5cd8c33-eac2-4620-8378-01de3c9ffb5f/Untitled.png)

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

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56a75b9d-8c60-47dc-86ae-2940317478a4/media_images_Validation_99_4.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56a75b9d-8c60-47dc-86ae-2940317478a4/media_images_Validation_99_4.jpg)

Labelled Image → Correctly labelled image from Open Images.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e26b46df-20bd-4077-b231-0d204808c8dd/media_images_Validation_99_5.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e26b46df-20bd-4077-b231-0d204808c8dd/media_images_Validation_99_5.jpg)

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

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6ba6bd11-5157-4632-8b6a-b6d2b8ada89b/media_images_Results_50_0.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6ba6bd11-5157-4632-8b6a-b6d2b8ada89b/media_images_Results_50_0.png)

Evaluation of model over time using Weights & Bias

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/470930dd-dff9-4670-a949-a242ce56b3e5/media_images_Results_50_1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/470930dd-dff9-4670-a949-a242ce56b3e5/media_images_Results_50_1.png)

Confusion Matrix

The confusion matrix shows how accurate the model is for individual classes based on what the actual label in the image is (x-axis) and what the model predicted (y-axis). The darker the box, the more accurate the model is for that particular class.

[Accuracy for each class:](https://www.notion.so/c3c630e1949d404fac6491e1ed633b89)

**Example Predictions:**

By using Weights and Bias we can look at some of the predictions the model made on a test batch. For each test batch the first image corresponds to the correctly labelled images, while the second image is the models prediction on what is in the image.

Looking at some of the predictions shows some interesting results. For example in test batch 3, the model appears to pick up an extra billiard table in the background that was missed in the labelled data (see far right image, second row in test batch 3). Also in test batch 3 it appears to have picked up some extra tv screens in the background (see bottom right image). Although the model did miss out some objects within an image, it's really interesting seeing the model pick up different objects not described in the labelled data. 

**Test batch 1:**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d3c6a705-2138-4f7e-9737-01e42c8fd11d/media_images_Validation_49_0.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d3c6a705-2138-4f7e-9737-01e42c8fd11d/media_images_Validation_49_0.jpg)

test_batch0_labels.jpg - Labelled images

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/314164bd-d861-4f21-8017-6c45a34469f4/media_images_Validation_49_1.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/314164bd-d861-4f21-8017-6c45a34469f4/media_images_Validation_49_1.jpg)

test_batch0_pred.jpg - Model prediction

**Test batch 2:**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/83ab6ba1-4a84-4e49-a5b8-dbe7f7ae2832/media_images_Validation_49_2.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/83ab6ba1-4a84-4e49-a5b8-dbe7f7ae2832/media_images_Validation_49_2.jpg)

test_batch1_labels.jpg - Labelled images

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/78b62268-31d0-49f2-97fb-cf9efebf51ea/media_images_Validation_49_3.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/78b62268-31d0-49f2-97fb-cf9efebf51ea/media_images_Validation_49_3.jpg)

test_batch1_pred.jpg - Model prediction

**Test batch 3**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1ea46cd2-2637-49d8-a0be-093daaa177a4/media_images_Validation_49_4.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1ea46cd2-2637-49d8-a0be-093daaa177a4/media_images_Validation_49_4.jpg)

test_batch2_labels.jpg - Labelled images

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2f8bcc4a-fd55-48b1-b2b8-2f684fbed150/media_images_Validation_49_5.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2f8bcc4a-fd55-48b1-b2b8-2f684fbed150/media_images_Validation_49_5.jpg)

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

Streamlit allows for us to bring our models to life by allowing for simple creation of a frontend of our model. Creating a frontend for our models is really important, as models tend to get left in the Jupyter notebook which doesn't allow for average person interaction. This is important to note as the majority of people using our models, whatever they may be won't know how to use Jupyter notebook. Plus, who doesn't like a cool frontend to look at. There's alternatives such as Flask [see here]([https://github.com/Pasquale-97/spotify-recommender](https://github.com/Pasquale-97/spotify-recommender)), however, in this scenario we will be using Streamlit. Before reading on, please checkout this [Github repo]([https://github.com/hassan-baydoun/python_final_project](https://github.com/hassan-baydoun/python_final_project)) that really helped implement the front end.

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

[Useful Resources](https://www.notion.so/93d808670134460c9833c77e668881f7)
