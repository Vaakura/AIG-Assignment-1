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

df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact,:day_of_week, :poutcome,:y]]




#store final data set with selcted columns "data"

cat_cols = df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact,:day_of_week, :poutcome,:y]]

#search for missing ddata on the "data" columns
#No missing data on all coumuns

describe(cat_cols)

#Increase colums size

ENV["COLUMNS"] = 1000

#encode categorical data in numeric
#Marital

convert_string(str)= try parse(Float64,str) catch; 
    if str == "married"
 return 0
end
if str == "single"
    return 1
end
    if str == "divorced"
    return 2
end
    if str == "unkown"
    return 3
end
return 4
    end;
    
    cat_cols.marital = map(convert_string,cat_cols.marital)

#encode categorical data in numeric
#job

convert_string(str)= try parse(Float64,str) catch; 
    if str == "housemaid"
 return 0
end
if str == "services"
    return 1
end
    if str == "admin."
    return 2
end
    if str == "blue-collar"
    return 3
end
      if str == "retired"
    return 4
end
     if str == "management"
    return 5
end
      if str == "technician"
    return 6
end
    if str == "self-employed"
    return 7
end
    if str == "unknown"
    return 8
end
    
return 9
    end;
    
    cat_cols.job = map(convert_string,cat_cols.job)

#encode categorical data in numeric
#education

convert_string(str)= try parse(Float64,str) catch; 
    if str == "basic.4y"
 return 0
end
if str == "high.school"
    return 1
end
    if str == "basic.6y"
    return 2
end
    if str == "basic.9y"
    return 3
end
      if str == "university.degree"
    return 4
end
       if str == "professional.course"
    return 5
end
           if str == "unknown"
    return 6
end
  
  
return 7
    end;
    
    cat_cols.education = map(convert_string,cat_cols.education)

#encode categorical data in numeric
#default

convert_string(str)= try parse(Float64,str) catch; 
    if str == "no"
 return 0
end
if str == "yes"
    return 1
end
    if str == "unknown"
    return 2
end 
return 3
    end;
    
    cat_cols.default = map(convert_string,cat_cols.default)

#encode categorical data in numeric
#housing

convert_string(str)= try parse(Float64,str) catch; 
    if str == "no"
 return 0
end
if str == "yes"
    return 1
end
    if str == "unknown"
    return 2
end 
return 3
    end;
    
    cat_cols.housing = map(convert_string,cat_cols.housing)

#encode categorical data in numeric
#loan

convert_string(str)= try parse(Float64,str) catch; 
    if str == "no"
 return 0
end
if str == "yes"
    return 1
end
    if str == "unknown"
    return 2
end 
return 3
    end;
    
    cat_cols.loan = map(convert_string,cat_cols.loan)

#encode categorical data in numeric
#contact

convert_string(str)= try parse(Float64,str) catch; 
    if str == "telephone"
 return 0
end
if str == "cellular"
    return 1
end
    if str == "unknown"
    return 2
end 
return 3
    end;
    
    cat_cols.contact = map(convert_string,cat_cols.contact)

#encode categorical data in numeric
#day_of_week

convert_string(str)= try parse(Float64,str) catch; 
    if str == "mon"
 return 0
end
if str == "tue"
    return 1
end
    if str == "wed"
    return 2
end 
      if str == "thu"
    return 3
end 
       if str == "fri"
    return 4
end 
return 5
    end;
    
    cat_cols.day_of_week = map(convert_string,cat_cols.day_of_week)

#encode categorical data in numeric
#poutcome

convert_string(str)= try parse(Float64,str) catch; 
    if str == "nonexistent"
 return 0
end
if str == "failure"
    return 1
end
    if str == "success"
    return 2
end 
return 5
    end;
    
    cat_cols.poutcome = map(convert_string,cat_cols.poutcome)

#encode categorical data in numeric
#y
convert_string(str)= try parse(Float64,str) catch; 
    if str == "no"
 return 0
end
if str == "yes"
    return 1
end
    if str == "unknown"
    return 3
end 
return 4
    end;
    
    cat_cols.y = map(convert_string,cat_cols.y)

#All categorcirial data encoded to numeric

cat_cols

#split data into testing and training

#Creating a data vector

x = cat_cols[:,1:11];
print(x);


#Create data vector for prediction  "y"

y = cat_cols[:,:11];
print(y);

#Create data matrix

xmat = convert(Matrix,x);
print(xmat);

#Dividing the matrix
#41189  -Total number of records
#20 testing  = 8,237.8‬
#80 training  = 32,951.2

 Xtesting = xmat[1: 8237,8‬, :]
 Xtrain = xmat[1: 32951,2, :]
 


