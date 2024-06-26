from NoPlotParquet import dataGenerator
from CompareParquet import StoreVariables

size_data = 5 #This is the span of every parameter, it will be raised to the 7th power
data = dataGenerator(size_data) #Generates a list of dictionaries containing all the simulated data, check NoPlotParquet for more information


df = pd.DataFrame(data) #Creates a dataframe to be easier to save

file_name = f'data_main'
folder_path = "C:\\Users\\Utente\\Desktop\\Trial"
file_path = os.path.join(folder_path, file_name)
open(file_path, 'w').close() #Cleans the file or creates it

df.to_parquet(file_path, engine='pyarrow', index=False) #Dump the data in the parquet file


#This reads back the file and fits to an example of experimental data exploiting the fitting parameters

param1, param2, param3, rot_i, rot_a, magnc, shift = StoreVariables(file_path)