## Work-from-Home Prediction

**Authors:** Aditi Shilke & Neha Jain  
**Course:** Machine Learning with Big Data at Frankfurt School of Finance and Management  
**Taught by:** Dr. Peter Roßbach  

## Project Overview
This project focuses on creating an **end-to-end solution** that bridges **machine learning with business context**. Starting with data provided by stakeholders, we processed, analyzed, and modeled it to generate **actionable insights**. Every decision—from imputing null values to hyperparameter tuning—was made with a clear rationale, explicitly presented in the attached PDF.

## Problem Statement
The aim is to **predict whether an employee will work from home on the next day**. Accurate forecasting helps companies plan resources such as office space. The dataset contains **9 features**:  

- **identifier:** Unique key of the data rows  
- **distance_office:** Distance in kilometers from the employee’s house to the workplace  
- **salary_range:** Range in euros of the employee’s yearly income  
- **gas_price:** Price of gas per liter near the employee’s residence  
- **public_transportation_cost:** Price in euros of public transportation from residence to workplace  
- **wfh_prev_workday:** Whether the employee worked from home the previous workday  
- **workday:** Day of the week for which we want to predict WFH  
- **tenure:** Number of years the employee has been at the company  
- **work_home:** Target variable, 1 if the employee worked from home that day, 0 otherwise  

## Approach
The project explored various **Supervised Machine Learning models** and explicitly evaluated each choice in terms of **business impact**, not just raw accuracy. The attached PDF outlines the reasoning behind each step and identifies the model best suited for this business case (**spoiler:** it is not the model with the highest test accuracy!).

## Outcome
- **Grade:** 1 (on a scale of 1 to 5, 1 being the best)  
- It sparked further interest in exploring the intersection of **data science, machine learning, and business impact!** :D
