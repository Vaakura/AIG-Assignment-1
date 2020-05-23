using Knet, DataFrames, Gadfly, Cairo, CSV, FreqTables, CategoricalArrays

# Loading or reading the file
df = readtable("C:\\Users\\Vaakura\\Desktop\\AIG2020\\bank-additional-full.csv", separator = ';');
# or with CSV.read from CSV (using CSV; CSV.read("file"))
println("# Data Description")
println(describe(df))



# Data cleaning

#Check inconsistant column names

head(df,4)

#Print out coulum names

names(df)

#Chage all strings to lowercase

lowercase(string(names(df)))

names!(df,[:age, :job, :marital, :education, :default, :housing, :loan, :contact, :month, :day_of_week, :duration, :campaign, :pdays, :previous, :poutcome, :emp_var_rate, :cons_price_idx, :cons_conf_idx, :euribor3m, :nr_employed, :y])

#Specifying a group of strings to be converted to NA values during reading:

df1 = readtable("C:\\Users\\Vaakura\\Desktop\\AIG2020\\bank-additional-full.csv",nastrings=["NA", "na", "n/a", "missing"])



#select all columns for testing

df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact, :y]]




#store final data setwith selcted coluns to varibal "testing"

testing = df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact, :y]]

#select all columns for testing

df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact, :y]]

#search for missing ddata on the "testing" columns

describe(testing)


#Increase colums size

ENV["COLUMNS"] = 1000

#represent categorical data in numeric

marital = df[:,[:marital,]]



marital = CategoricalArray(["married", "single", "divorced", "unknown"], ordered=true) 

#call levels function to order 

levels(marital)

marital[1]


marital[2]



marital[3]

marital[4]

marital[1] = "single"

marital[2] = "married"

marital[3] = "divorced"

marital[4] = "unknown"



