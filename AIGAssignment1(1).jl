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




#store final data set with selcted columns "data"

data2 = df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact, :y]]

#select all columns for testing

df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact, :y]]

#search for missing ddata on the "data" columns

describe(data2)


#Increase colums size

ENV["COLUMNS"] = 1000

#represent categorical data in numeric

marital = df[:,[:marital,]]

marital = df[:,[:marital,]]

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
    
    data2.marital = map(convert_string,data2.marital)

job = df[:,[:job,]]


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
    
    data2.job = map(convert_string,data2.job)

education = df[:,[:education,]]

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
  
  
return 6
    end;
    
    data2.education = map(convert_string,data2.education)

default = df[:,[:default,]]

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
  
  
return 6
    end;
    
    data2.education = map(convert_string,data2.education)

housing = df[:,[:housing,]]

marital[3] = "single"

loan = df[:,[:loan,]]

data


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
return missing
    end;
    
    test1.marital = map(convert_string,test1.marital)

test1

test1 = df[:,[:age,:job,:marital, :education, :default, :housing, :loan,:contact, :y]]
