"""
data_model.py — ChurnOps

Pydantic request model for the POST /predict endpoint.
All categorical fields use Literal types to prevent invalid values
reaching the preprocessor.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Customer(BaseModel):
    """
    Customer feature payload for churn prediction.

    Categorical fields match the exact values seen during training.
    Numerical fields have realistic min/max bounds for basic sanity checking.
    """

    # ---- Demographics -------------------------------------------------------
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=0, le=120)
    Married: Literal["Yes", "No"]
    Number_of_Dependents: int = Field(..., ge=0, le=20, alias="Number of Dependents")
    Satisfaction_Score: int = Field(..., ge=1, le=5, alias="Satisfaction Score")

    # ---- Services -----------------------------------------------------------
    Number_of_Referrals: int = Field(0, ge=0, alias="Number of Referrals")
    Tenure_in_Months: int = Field(..., ge=0, le=120, alias="Tenure in Months")
    Offer: Literal["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
    Phone_Service: Literal["Yes", "No"] = Field(..., alias="Phone Service")
    Multiple_Lines: Literal["Yes", "No"] = Field(..., alias="Multiple Lines")
    Internet_Service: Literal["Yes", "No"] = Field(..., alias="Internet Service")
    Internet_Type: Literal["None", "DSL", "Cable", "Fiber Optic"] = Field(..., alias="Internet Type")
    Online_Security: Literal["Yes", "No"] = Field(..., alias="Online Security")
    Online_Backup: Literal["Yes", "No"] = Field(..., alias="Online Backup")
    Device_Protection_Plan: Literal["Yes", "No"] = Field(..., alias="Device Protection Plan")
    Premium_Tech_Support: Literal["Yes", "No"] = Field(..., alias="Premium Tech Support")
    Streaming_TV: Literal["Yes", "No"] = Field(..., alias="Streaming TV")
    Streaming_Movies: Literal["Yes", "No"] = Field(..., alias="Streaming Movies")
    Streaming_Music: Literal["Yes", "No"] = Field(..., alias="Streaming Music")
    Unlimited_Data: Literal["Yes", "No"] = Field(..., alias="Unlimited Data")
    Referred_a_Friend: Literal["Yes", "No"] = Field(..., alias="Referred a Friend")

    # ---- Contract / Billing -------------------------------------------------
    Contract: Literal["Month-to-Month", "One Year", "Two Year"]
    Paperless_Billing: Literal["Yes", "No"] = Field(..., alias="Paperless Billing")
    Payment_Method: Literal["Bank Withdrawal", "Credit Card", "Mailed Check"] = Field(
        ..., alias="Payment Method"
    )

    # ---- Charges ------------------------------------------------------------
    Avg_Monthly_Long_Distance_Charges: float = Field(0.0, ge=0.0, alias="Avg Monthly Long Distance Charges")
    Avg_Monthly_GB_Download: float = Field(0.0, ge=0.0, alias="Avg Monthly GB Download")
    Monthly_Charge: float = Field(..., ge=0.0, alias="Monthly Charge")
    Total_Charges: float = Field(..., ge=0.0, alias="Total Charges")
    Total_Refunds: float = Field(0.0, ge=0.0, alias="Total Refunds")
    Total_Extra_Data_Charges: float = Field(0.0, ge=0.0, alias="Total Extra Data Charges")
    Total_Long_Distance_Charges: float = Field(0.0, ge=0.0, alias="Total Long Distance Charges")
    Total_Revenue: float = Field(..., ge=0.0, alias="Total Revenue")

    model_config = {"populate_by_name": True}