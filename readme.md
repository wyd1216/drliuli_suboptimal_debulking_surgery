## Data preprocess before train the model

### Dataset information staticstics.

1. Dataset information staticstics.

   - Directory: DataPreprocess

   - `dataset_statistics.ipynb`
   
     The information of 'PID' (labeled by doctor), 'PID1' (labeled by CT), 'RD' (size of leision), 'img_type' (used for features extraction), 'image' (path), 'mask'
     (path), 'series' (sequence ID), 'label' (positive or negative).
     
   - output data.

     + `icc_data_info.csv`
       
       Used for check consistency.

     + `all_data_info.csv'

       The general data information.

2. Radiomics features extraction.

  - file: `radiomics_feature_extraction/radiomics_extraction.py`

  - input data:

    step 1 output.
    
  - Run `radiomics_extraction.py` to get the features files. (Only 'image' and 'mask' property remained)

3. Train and test dataset split

  - file: `DataPreprocess/split.ipynb`

4. Features filter.

 - file: `Feature_filter/Feature_extraction.ipynb`

 - Then get the same features for test data,

   file: `Feature_filter/Feature_extraction_test.ipynb`

5. Train
    
  - directory: `Train`

  Train the data for different model and show the results.


6. paper_data

   Prepare the figure for the paper.

   - `./paper_figure_draw.ipynb`
   
   - `paper_data`
   
 
7. paper_modify based review opnion

   - `20230108`
   