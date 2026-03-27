import streamlit as st
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- YOUR ORIGINAL BACKEND CODE ---
df = sns.load_dataset('penguins').dropna()

# --- UI ---
st.title("Penguin Dataset - MANOVA & ANOVA Analysis")

st.sidebar.header("Select Analysis")
analysis = st.sidebar.radio("Analysis Type", ["MANOVA", "Individual ANOVA"])
factor = st.sidebar.selectbox("Select Factor", ["island", "sex", "species"])
dependent_vars = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]

# --- MANOVA ---
if analysis == "MANOVA":
    st.subheader(f"MANOVA — {factor}")

    if st.button("Run MANOVA"):
        # YOUR ORIGINAL CODE ✅
        maov = MANOVA.from_formula(
            f'bill_length_mm + bill_depth_mm + flipper_length_mm ~ {factor}', df)
        result = maov.mv_test()
        st.text(str(result))

        st.write("### Interpretation")
        st.info(f"If all 4 test statistics (Wilks, Pillai, Hotelling, Roy) "
                f"have p < 0.05 → **{factor}** significantly affects "
                f"the dependent variables combined.")

# --- INDIVIDUAL ANOVA + TUKEY ---
elif analysis == "Individual ANOVA":
    dep_var = st.sidebar.selectbox("Select Dependent Variable", dependent_vars)
    st.subheader(f"One-Way ANOVA — {dep_var} ~ {factor}")

    if st.button("Run ANOVA"):
        # YOUR ORIGINAL CODE ✅
        fit = ols(f'{dep_var} ~ {factor}', df).fit()
        result = sm.stats.anova_lm(fit)

        st.write("### ANOVA Table")
        st.dataframe(result)

        # YOUR ORIGINAL CODE ✅
        tukey = pairwise_tukeyhsd(df[dep_var], groups=df[factor])
        st.write("### Tukey HSD")
        st.dataframe(pd.DataFrame(
            tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        ))

        st.write("### Group Means")
        st.dataframe(df.groupby(factor)[dep_var].mean()
                       .sort_values(ascending=False)
                       .reset_index())

        # interpretation
        p = result.loc[factor, 'PR(>F)']
        if p < 0.05:
            st.success(f"**{factor}** significantly affects "
                      f"**{dep_var}** (p = {p:.6f}) ✅")
        else:
            st.error(f"**{factor}** does NOT significantly affect "
                    f"**{dep_var}** (p = {p:.6f}) ❌")