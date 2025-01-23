## Label Studio Setup and Usage

### 1. Navigate to the Repository

Open your terminal and change to the repository root directory:

```shell
cd just_wash_your_hands
````

### 2. Generate Label Studio JSON

Use [this script](https://github.com/i-petrychko/just_wash_your_hands/tree/main/preprocessing/label_studio/operations.py)


### 3. Set Up Access to Local Data

Run the following commands to give Label Studio access to your local files:

```shell
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
```

### 4. Start Label Studio

```shell
label-studio start
```

### 5. Create a Project in Label Studio

Open Label Studio in your browser (it should automatically open when you run label-studio start).
Create a new project and configure the settings.

### 6. Set Up Cloud Storage
When prompted, choose local storage for your project.

### 7. Set Up the Labeling Interface
Use the labeling interface XML provided in the following file:

[Labeling Interface XML](https://github.com/i-petrychko/just_wash_your_hands/tree/main/preprocessing/label_studio/labeling_interface.xml)

This defines how the images and labels are presented during the labeling process.

### 8. Import the Generated JSON
Finally, import the generated label_studio.json into your project to start the labeling process.

