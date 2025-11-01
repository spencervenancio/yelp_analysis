## Download the data

The Yelp Open Dataset can be downloaded from [yelp.com](https://business.yelp.com/data/resources/open-dataset/)

Scroll down and select `Download JSON`. 

Expand the `.zip` and then the `.tar` file. Then move the `yelp_academic_dataset_<TABLE>.json` files to the `data/raw/` directory. 

## Installing Dependencies
First check that
```
(base) spencervenancio@Spencers-Air yelp_analysis % pwd 
../../yelp_analysis
```
Then you can install the necessary dependencies by running in your terminal
```
make requirements
```

## Setting Up Kernal
You should also set up a `Jupyter` kernal so you can run `.ipynb` files using the `uv` requirements. In your terminal run: 
```
uv add --dev ipykernel
```
Then select `src` as the kernal when you first open up `.ipynb` files 
