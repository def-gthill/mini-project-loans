"""
Model components that need to be picklable.
"""

def create_total_income(df):
    return df.assign(TotalIncome=(df.ApplicantIncome + df.CoapplicantIncome))


def combine_income(df):
    return create_total_income(df).drop(
        ['ApplicantIncome', 'CoapplicantIncome'], axis=1
    )
