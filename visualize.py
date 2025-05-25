import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(features):
    sns.pairplot(features, hue='cluster')
    plt.title('User Clusters')
    plt.show()

def show_rules(rules):
    return rules[['antecedents', 'consequents', 'support', 'confidence']]
