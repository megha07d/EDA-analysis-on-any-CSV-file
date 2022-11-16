# Meghana - CS20B1060
# Madhav - CS20B1047
# Web Application to perform Statistical Analysis, EDA & Data Visualisation

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

class DataFrame_Loader():

	
	def __init__(self):
		
		print("Loadind DataFrame")
		
	def read_csv(self,data):
		self.df = pd.read_csv(data)
		return self.df

class EDA_Dataframe_Analysis():

	
	def __init__(self):
		
		print("General_EDA object created")

	def show_dtypes(self,x):
		return x.dtypes.astype(str)

	def show_columns(self,x):
		return x.columns

	def show_hist(self,x):
		return x.hist()

	def Show_Missing(self,x):
		y=x.isna().sum()
		return y
	def show_hist(self,x):
		return x.hist()
	
	def Tabulation(self,x):
		table = pd.DataFrame(x.dtypes,columns=['dtypes'])
		table1 =pd.DataFrame(x.columns,columns=['Names'])
		table = table.reset_index()
		table['No of Missing'] = x.isnull().sum().values
		table['No of Uniques'] = x.nunique().values
		table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
		table['First Observation'] = x.loc[0].values
		table['Second Observation'] = x.loc[1].values
		table['Third Observation'] = x.loc[2].values
		return table.astype(str)
	
	def Numerical_variables(self,x):
		Num_var = [var for var in x.columns if x[var].dtypes!="object"]
		Num_var = x[Num_var]
		return Num_var

	def categorical_variables(self,x):
		cat_var = [var for var in x.columns if x[var].dtypes=="object"]
		cat_var = x[cat_var]
		return cat_var
	
	def impute(self,x):
		df=x.dropna()
		return df

class Attribute_Information():

	def __init__(self):
		
		print("Attribute Information object created")
		
	def Column_information(self,data):
	
		data_info = pd.DataFrame(
								columns=['No of observation',
										'No of Variables',
										'No of Numerical Variables',
										'No of Factor Variables',
										'No of Categorical Variables',
										'No of Logical Variables',
										'No of Date Variables',
										'No of zero variance variables'])


		data_info.loc[0,'No of observation'] = data.shape[0]
		data_info.loc[0,'No of Variables'] = data.shape[1]
		data_info.loc[0,'No of Numerical Variables'] = data._get_numeric_data().shape[1]
		data_info.loc[0,'No of Factor Variables'] = data.select_dtypes(include='category').shape[1]
		data_info.loc[0,'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
		data_info.loc[0,'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
		data_info.loc[0,'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
		data_info.loc[0,'No of zero variance variables'] = data.loc[:,data.apply(pd.Series.nunique)==1].shape[1]

		data_info =data_info.transpose()
		data_info.columns=['value']
		data_info['value'] = data_info['value'].astype(int)


		return data_info
	
	def __get_missing_values(self,data):
		
		#Getting sum of missing values for each feature
		missing_values = data.isnull().sum()
		#Feature missing values are sorted from few to many
		missing_values.sort_values(ascending=False, inplace=True)
		
		#Returning missing values
		return missing_values
	
	def __iqr(self,x):
		return x.quantile(q=0.75) - x.quantile(q=0.25)

	def __outlier_count(self,x):
		upper_out = x.quantile(q=0.75) + 1.5 * self.__iqr(x)
		lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
		return len(x[x > upper_out]) + len(x[x < lower_out])

	def num_count_summary(self,df):
		df_num = df._get_numeric_data()
		data_info_num = pd.DataFrame()
		i=0
		for c in  df_num.columns:
			data_info_num.loc[c,'Negative values count']= df_num[df_num[c]<0].shape[0]
			data_info_num.loc[c,'Positive values count']= df_num[df_num[c]>0].shape[0]
			data_info_num.loc[c,'Zero count']= df_num[df_num[c]==0].shape[0]
			data_info_num.loc[c,'Unique count']= len(df_num[c].unique())
			data_info_num.loc[c,'Negative Infinity count']= df_num[df_num[c]== -np.inf].shape[0]
			data_info_num.loc[c,'Positive Infinity count']= df_num[df_num[c]== np.inf].shape[0]
			data_info_num.loc[c,'Missing Percentage']= df_num[df_num[c].isnull()].shape[0]/ df_num.shape[0]
			data_info_num.loc[c,'Count of outliers']= self.__outlier_count(df_num[c])
			i = i+1
		return data_info_num
	
	def statistical_summary(self,df):
	
		df_num = df._get_numeric_data()

		data_stat_num = pd.DataFrame()

		try:
			data_stat_num = pd.concat([df_num.describe().transpose(),
									   pd.DataFrame(df_num.quantile(q=0.10)),
									   pd.DataFrame(df_num.quantile(q=0.90)),
									   pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
			data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
		except:
			pass

		return data_stat_num



	
def main():
    
	st.title("Summary Statistics, EDA & Data Visualisation")
	
	# st.info("Upload your csv file here ;)") 

	st.subheader("Exploratory Data Analysis")

	data = st.file_uploader("Upload your dataset here", type=["csv"])
	
	if data is not None :
			
		df = load.read_csv(data)
			
		st.success("CSV File Loaded successfully")

		listy1 = ['Select','Data Preview','Data Types','Columns','Number of missing values','Column Information','Show Selected Columns']

		listy2=['Select','Aggregation Tabulation','Number Count Summary','Statistical Summary','Numerical Variables','Categorical Variables','DropNA']
		option1 = st.sidebar.selectbox('Understanding Data',listy1)
		option2 = st.sidebar.selectbox('Analysis of data',listy2)

		if option1==listy1[1]:
			st.subheader("Preview of your data : ")
			st.dataframe(df.head())

		if option1==listy1[2]:
			st.subheader("Data Types : ")
			st.write(dataframe.show_dtypes(df))

		if option1==listy1[3]:
			st.subheader("Columns in the dataset : ")
			st.write(dataframe.show_columns(df))

		if option1==listy1[4]:
			st.subheader("Number of missing values : ")
			st.write(dataframe.Show_Missing(df))

		if option1==listy1[5]:
			st.subheader("Column Information : ")
			st.write(info.Column_information(df))

		if option1==listy1[6]:
			selected_columns = st.multiselect("Select Columns :",dataframe.show_columns(df))
			st.subheader("Selected columns : ")   
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if option2==listy2[1]:
			st.subheader("Aggregation Table : ")
			st.write(dataframe.Tabulation(df))

		if option2==listy2[2]:
			st.subheader("Number Count Summary : ")
			st.write(info.num_count_summary(df))

		if option2==listy2[3]:
			st.subheader("Statistical Analysis : ")
			st.write(info.statistical_summary(df))	

		if option2==listy2[4]:
			st.subheader("Numerical Variables : ")
			num_df = dataframe.Numerical_variables(df)
			numer_df=pd.DataFrame(num_df)                
			st.dataframe(numer_df)

		if option2==listy2[5]:
			st.subheader("Categorical Variables : ")
			new_df = dataframe.categorical_variables(df)
			catego_df=pd.DataFrame(new_df)                
			st.dataframe(catego_df)

		if option2==listy2[6]:
			st.subheader("Drop NA : ")
			num_df = dataframe.Numerical_variables(df)
			imp_df = dataframe.impute(num_df)
			st.dataframe(imp_df)
			st.download_button(
				label="Download data as CSV",
				data=imp_df.to_csv(),
				file_name='removed_Na.csv',
				mime='text/csv',
			)

		listy3 = ['Select','Histogram','Bar graph','Box plot','Scatter plot','Dist plot','Frequency Distribution']
		option3 = st.sidebar.selectbox('Univariate Analysis',listy3)

		listy4 = ['Select','Bivariate Scattering','Heatmap','Multivariate']
		option4 = st.sidebar.selectbox('Multivariate Analysis',listy4)

			
		if option3==listy3[1]:
			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Column for Histogram :",all_columns_names)
			st.write(dataframe.show_hist(df[selected_columns_names]))
			st.pyplot()
			
		if option3==listy3[5]:
			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns Distplot :",all_columns_names)
			st.write(dataframe.Show_DisPlot(df[selected_columns_names]))
			st.pyplot()



if __name__ == '__main__':
	load = DataFrame_Loader()
	dataframe = EDA_Dataframe_Analysis()
	info = Attribute_Information()
	main()





