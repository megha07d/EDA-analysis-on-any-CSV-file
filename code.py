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
	st.title("Machine Learning Application for Automated EDA")
	
	st.info("text line")
	"""okie""" 
	activities = ["General EDA","EDA For Linear Models","Model Building for Classification Problem"]	
	choice = st.sidebar.selectbox("Select Activities",activities)


	if choice == 'General EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv"])
		if data is not None:
			df = load.read_csv(data)
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")
			

			if st.checkbox("Show dtypes"):
				st.write(dataframe.show_dtypes(df))

			if st.checkbox("Show Columns"):
				st.write(dataframe.show_columns(df))

			if st.checkbox("Show Missing"):
				st.write(dataframe.Show_Missing(df))

			if st.checkbox("column information"):
				st.write(info.Column_information(df))

			if st.checkbox("Aggregation Tabulation"):
				st.write(dataframe.Tabulation(df))

			if st.checkbox("Num Count Summary"):
				st.write(info.num_count_summary(df))

			if st.checkbox("Statistical Summary"):
				st.write(info.statistical_summary(df))	

			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",dataframe.show_columns(df))
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Numerical Variables"):
				num_df = dataframe.Numerical_variables(df)
				numer_df=pd.DataFrame(num_df)                
				st.dataframe(numer_df)

			if st.checkbox("Categorical Variables"):
				new_df = dataframe.categorical_variables(df)
				catego_df=pd.DataFrame(new_df)                
				st.dataframe(catego_df)

			if st.checkbox("DropNA"):
				num_df = dataframe.Numerical_variables(df)
				imp_df = dataframe.impute(num_df)
				st.dataframe(imp_df)

			if st.checkbox("Missing after DropNA"):
				num_df = dataframe.Numerical_variables(df)
				imp_df = dataframe.impute(num_df)
				st.write(dataframe.Show_Missing(imp_df))

			st.subheader("UNIVARIATE ANALYSIS")
			
			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Column for Histogram ",all_columns_names)
			if st.checkbox("Show Histogram for Selected variable"):
				st.write(dataframe.show_hist(df[selected_columns_names]))
				st.pyplot()

			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns Distplot ",all_columns_names)
			if st.checkbox("Show DisPlot for Selected variable"):
				st.write(dataframe.Show_DisPlot(df[selected_columns_names]))
				st.pyplot()

			


if __name__ == '__main__':
	load = DataFrame_Loader()
	dataframe = EDA_Dataframe_Analysis()
	info = Attribute_Information()
	main()





