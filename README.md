# binde
## Library for Differential expression analysis of binary appearance patterns

- "binary_de_utils.py" - Library with Appearance Differential Expression analysis functions + Toy Usage example
- "analysis_utils.py"  - Library with auxiliary functions
- "binary_analysis_test.py" - Real-Data usage example of Appearance Differential Expression analysis functions for article "Differential expression analysis of binary appearance patterns"
- "de_vs_binary.input.pkl" - Input data (based on "Proteome sampling with e-biopsy enables differentiation between cutaneous squamous cell carcinoma and basal cell carcinoma", https://www.medrxiv.org/content/10.1101/2022.12.22.22283845v1)
- "stdout.log" - Expected console output

# Toy Usage Example output:
<p><i>Proportion-based test (zscore,pvalue)</i><br>
- (32/50 vs 18/60 successes), alternative=two-sided: (array(3.56595008), array(0.00036254))<br>
- (18/60 vs 32/50 successes), alternative=two-sided: (array(-3.56595008), array(0.00036254))<br>
- (32/50 vs 18/60 successes), alternative=less: (array(3.56595008), array(0.99981873))<br>
- (32/50 vs 18/60 successes), alternative=greater: (array(3.56595008), array(0.00018127))<br>
</p>
<p><i>HG-based test (pvalue)</i><br>
- (32/50 vs 18/60 successes): 0.0003407990265263909<br>
- (18/60 vs 32/50 successes): 0.9999253039889312<br>
</p>
