using Knet, DataFrames, Gadfly, Cairo, CSV, FreqTables, CategoricalArrays, Statistics, Plots

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

y_train = cat_cols[:,:11];
print(y_train);

y_test = cat_cols[:,:11];
print(y_test);

#Create data matrix

xmat = convert(Matrix,x);
print(xmat);

#Dividing the matrix fortesting and training


#41189  -Total number of records
#20% testing  = 8,237.8‬

Xtrain = xmat[1: 8238, :]

#41189  -Total number of records
#80% training  = 32,951.2

Xtesting = xmat[1: 32951, :]

# Normalise the training design matrix
"""
This function attempts to standardise the design matrix (X) user pass.
The input data X is standardised and return along with other learned metrics.
A tuple is returned representing (Standardised data, mean, std deviation).
"""
function scale_features(X)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)

    X_norm = (X .- μ) ./ σ

    return (X_norm, μ, σ);
end


# Normalise the testing design matrix
"""
This functions uses the mean and standard deviation values users pass to
normalise a new design matrix.
"""
function transform_features(X, μ, σ)
    X_norm = (X .- μ) ./ σ
    return X_norm;
end


# Scale training features and get artificats for future use
X_train_scaled, μ, σ = scale_features(Xtrain);

# Transform the testing features by using the learned artifacts
X_test_scaled = transform_features(Xtesting, μ, σ);

"""
This function applies the sigmoid activation to any supplied scalar/vector.
"""
function sigmoid(z)
    return 1 ./ (1 .+ exp.(.-z))
end


"""
#The regularised cost function computes the batch cost with a lambda penalty (λ) as well.
The batch cost vector (𝐉) and the gradients (∇𝐉) of this vector are returned as a tuple.
"""
function regularised_cost(X, y, θ, λ)
    m = length(y)

    # Sigmoid predictions at current batch
    h = sigmoid(X * θ)

    # left side of the cost function
    positive_class_cost = ((-y)' * log.(h))

    # right side of the cost function
    negative_class_cost = ((1 .- y)' * log.(1 .- h))

    # lambda effect
    lambda_regularization = (λ/(2*m) * sum(θ[2 : end] .^ 2))

    # Current batch cost. Basically mean of the batch cost plus regularization penalty
    𝐉 = (1/m) * (positive_class_cost - negative_class_cost) + lambda_regularization

    # Gradients for all the theta members with regularization except the constant
    ∇𝐉 = (1/m) * (X') * (h-y) + ((1/m) * (λ * θ))  # Penalise all members

    ∇𝐉[1] = (1/m) * (X[:, 1])' * (h-y) # Exclude the constant

    return (𝐉, ∇𝐉)
end

"""
This function uses gradient descent to search for the weights 
that minimises the logit cost function.
A tuple with learned weights vector (θ) and the cost vector (𝐉) 
are returned.
"""
function logistic_regression_sgd(X, y, λ, fit_intercept=true, η=0.01, max_iter=1000)
    
    # Initialize some useful values
    m = length(y); # number of training examples

    if fit_intercept
        # Add a constant of 1s if fit_intercept is specified
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X # Assume user added constants
    end

    # Use the number of features to initialise the theta θ vector
    n = size(X)[2]
    θ = zeros(n)

    # Initialise the cost vector based on the number of iterations
    𝐉 = zeros(max_iter)

    for iter in range(1, stop=max_iter)

        # Calcaluate the cost and gradient (∇𝐉) for each iter
        𝐉[iter], ∇𝐉 = regularised_cost(X, y, θ, λ)

        # Update θ using gradients (∇𝐉) for direction and (η) for the magnitude of steps in that direction
        θ = θ - (η * ∇𝐉)
    end

    return (θ, 𝐉)
end



# Use gradient descent to search for the optimal weights (θ)
θ, 𝐉 = logistic_regression_sgd(X_train_scaled, y_train, 0.0001, true, 0.3, 3000);

# Plot the cost vector
#plot(𝐉, color="blue", title="Cost Per Iteration", legend=false,
 #    xlabel="Num of iterations", ylabel="Cost")

"""
This function uses the learned weights (θ) to make new predictions.
Predicted probabilities are returned.
"""
function predict_proba(X, θ, fit_intercept=true)
    m = size(X)[1]

    if fit_intercept
        # Add a constant of 1s if fit_intercept is specified
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X
    end

    h = sigmoid(X * θ)
    return h
end


"""
This function binarizes predicted probabilities using a threshold.
Default threshold is set to 0.5
"""
function predict_class(proba, threshold=0.5)
    return proba .>= threshold
end


# Training and validation score
train_score = mean(y_train .== predict_class(predict_proba(X_train_scaled, θ)));
test_score = mean(y_test .== predict_class(predict_proba(X_test_scaled, θ)));

# Training and validation score rounded to 4 decimals
println("Training score: ", round(train_score, sigdigits=4))
println("Testing score: ", round(test_score, sigdigits=4))

function confusionmatrix(y_train, Xtrain, d)
    c =zeros(d,d)
    for i in 1:length(Xtrain)
        c[Xtrain[i]+ 1, y_train[i] + 1]+= 1
    end
    return c
end

function accuracy(Xtrain, y_train)
    sum(y_train .== Xtrain)/length(Xtrain)
        end


# a function to measure accuracy
Accuracy(w, Xtrain, y_test) = sum((sign.(predict(w, Xtrain)) + 1) / 2 .== y_test) / length(y_test);

println(sum(y_train) / length(y_train), " , ", sum(y_test) / length(y_test))
println("")
println("# size of train and test data")
println(size(Xtrain), size(Xtesting))
println("")



function precision(y_train, Xtrain) 
    #tp = y_test
   # fp = Xtesting 
    for i in 1:length(Xtrain)
      if Xtrain[i] == 1 
          y_test += y_train[i]
      else 
          Xtesting  += y_train[i]
      end    
    end 
    return y_test / (y_test + Xtesting)
end


    






