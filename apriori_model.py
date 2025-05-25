from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def run_apriori(session_data, min_support=0.01, min_confidence=0.3):
    te = TransactionEncoder()
    te_ary = te.fit(session_data['product_id']).transform(session_data['product_id'])
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return rules
