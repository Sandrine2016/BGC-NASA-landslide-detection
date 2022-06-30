# BGC-NASA-landslide-detection
Detect landslide location, date, and category automatically.

1. Install docker application
2. Download landslides.tar and user files from [here](https://drive.google.com/drive/folders/1jpARrfLu9sGGVa9YCxBKmeEF0UJtMrMk) to your local directory
3. You can choose model and start-end date by editing the config file
4. Go to the directory where you save the landslides.tar in terminal
5. Type and run "docker load < landslides.tar" in  terminal
   The terminal should show Loaded image: landslides:latest
6. Type and run "docker run -v $(pwd)/user:/user landslides" in terminal
7. You can see the final results in user folder

Thanks!!
