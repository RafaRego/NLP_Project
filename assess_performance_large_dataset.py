# assess_performance_large.py
# Purpose: Evaluate sentiment model performance against recommendations on large dataset

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the scored data - CHANGED
df = pd.read_csv("large_dataset_scored.csv")

print("=" * 60)
print("SENTIMENT MODEL PERFORMANCE ASSESSMENT")
print("=" * 60)

# Basic statistics
print(f"\nTotal reviews: {len(df)}")
print(f"Predicted positive: {df['pred_label'].sum()} ({100*df['pred_label'].mean():.1f}%)")
print(f"Predicted negative: {len(df) - df['pred_label'].sum()} ({100*(1-df['pred_label'].mean()):.1f}%)")

# Check if 'label_bin' column exists - CHANGED (was 'recommended')
if 'label_bin' in df.columns:
    # Use label_bin directly (already binary 0/1) - CHANGED
    df_clean = df.dropna(subset=['label_bin'])
    
    print(f"\nReviews with recommendation data: {len(df_clean)}")
    print(f"Recommended: {df_clean['label_bin'].sum()} ({100*df_clean['label_bin'].mean():.1f}%)")
    print(f"Not recommended: {len(df_clean) - df_clean['label_bin'].sum()} ({100*(1-df_clean['label_bin'].mean()):.1f}%)")
    
    # Calculate accuracy - CHANGED (use label_bin instead of recommended_binary)
    accuracy = accuracy_score(df_clean['label_bin'], df_clean['pred_label'])
    print(f"\n{'='*60}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"{'='*60}")
    
    # Confusion Matrix - CHANGED
    cm = confusion_matrix(df_clean['label_bin'], df_clean['pred_label'])
    print("\nConfusion Matrix:")
    print("                 Predicted Negative  Predicted Positive")
    print(f"Actually Negative        {cm[0,0]:6d}              {cm[0,1]:6d}")
    print(f"Actually Positive        {cm[1,0]:6d}              {cm[1,1]:6d}")
    
    # Classification Report - CHANGED
    print("\nDetailed Classification Report:")
    print(classification_report(
        df_clean['label_bin'], 
        df_clean['pred_label'],
        target_names=['Not Recommended', 'Recommended']
    ))
    
    # Agreement analysis - CHANGED
    agreement = (df_clean['label_bin'] == df_clean['pred_label']).sum()
    print(f"\nAgreement: {agreement}/{len(df_clean)} ({100*agreement/len(df_clean):.1f}%)")
    print(f"Disagreement: {len(df_clean)-agreement}/{len(df_clean)} ({100*(len(df_clean)-agreement)/len(df_clean):.1f}%)")
    
    # Analyze disagreements - CHANGED
    disagreements = df_clean[df_clean['label_bin'] != df_clean['pred_label']]
    if len(disagreements) > 0:
        print(f"\n{'='*60}")
        print("DISAGREEMENT ANALYSIS")
        print(f"{'='*60}")
        
        false_positives = disagreements[disagreements['pred_label'] == 1]
        false_negatives = disagreements[disagreements['pred_label'] == 0]
        
        print(f"\nFalse Positives (predicted positive, but not recommended): {len(false_positives)}")
        print(f"False Negatives (predicted negative, but was recommended): {len(false_negatives)}")
        
        # Show confidence scores for disagreements
        print(f"\nAverage confidence for false positives: {false_positives['pred_prob_pos'].mean():.3f}")
        print(f"\nAverage confidence for false negatives: {false_negatives['pred_prob_pos'].mean():.3f}")
    
    # Visualize confusion matrix - CHANGED filename
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actually Not Recommended', 'Actually Recommended'])
    plt.title('Confusion Matrix: Sentiment vs Recommendation (Large Dataset)')
    plt.ylabel('Actual (Recommended)')
    plt.xlabel('Predicted (Sentiment)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_large.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix_large.png'")
    
    # Confidence distribution - CHANGED
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # By prediction
    axes[0].hist([df_clean[df_clean['pred_label']==0]['pred_prob_pos'],
                  df_clean[df_clean['pred_label']==1]['pred_prob_pos']], 
                 bins=20, label=['Predicted Negative', 'Predicted Positive'], alpha=0.7)
    axes[0].set_xlabel('Confidence Score (prob of positive)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution by Prediction')
    axes[0].legend()
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=1)
    
    # By actual recommendation - CHANGED
    axes[1].hist([df_clean[df_clean['label_bin']==0]['pred_prob_pos'],
                  df_clean[df_clean['label_bin']==1]['pred_prob_pos']], 
                 bins=20, label=['Not Recommended', 'Recommended'], alpha=0.7, color=['red', 'green'])
    axes[1].set_xlabel('Confidence Score (prob of positive)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution by Actual Recommendation')
    axes[1].legend()
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('confidence_distributions_large.png', dpi=300, bbox_inches='tight')
    print("Confidence distributions saved as 'confidence_distributions_large.png'")
    
    # Sample some disagreements - CHANGED
    if len(disagreements) > 0:
        print(f"\n{'='*60}")
        print("SAMPLE DISAGREEMENTS (first 5)")
        print(f"{'='*60}")
        for idx, row in disagreements.head(5).iterrows():
            print(f"\n--- Review {idx} ---")
            print(f"Text: {row['text_clean'][:200]}...")  # CHANGED to text_clean
            print(f"Recommended: {'Yes' if row['label_bin']==1 else 'No'}")  # CHANGED
            print(f"Predicted: {'Positive' if row['pred_label']==1 else 'Negative'} (confidence: {row['pred_prob_pos']:.3f})")
            if 'OverallScore' in df.columns:  # CHANGED column name
                print(f"Overall Score: {row['OverallScore']}")

else:
    print("\n⚠️  'label_bin' column not found in the dataset!")
    print("Available columns:", df.columns.tolist())

# Additional analysis with OverallScore if available - CHANGED
if 'OverallScore' in df.columns:
    print(f"\n{'='*60}")
    print("SENTIMENT vs OVERALL SCORE")
    print(f"{'='*60}")
    
    df_rating = df.dropna(subset=['OverallScore'])
    
    # Average rating by sentiment
    print(f"\nAverage overall score for predicted negative: {df_rating[df_rating['pred_label']==0]['OverallScore'].mean():.2f}")
    print(f"\nAverage overall score for predicted positive: {df_rating[df_rating['pred_label']==1]['OverallScore'].mean():.2f}")
    
    # Correlation
    correlation = df_rating['pred_prob_pos'].corr(df_rating['OverallScore'])
    print(f"\nCorrelation between confidence score and overall score: {correlation:.3f}")

print(f"\n{'='*60}")
print("ASSESSMENT COMPLETE")
print(f"{'='*60}")