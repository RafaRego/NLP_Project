# assess_performance.py
# Purpose: Evaluate sentiment model performance against recommendations

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the scored data
df = pd.read_csv("skytrax_subset_scored.csv")

print("=" * 60)
print("SENTIMENT MODEL PERFORMANCE ASSESSMENT")
print("=" * 60)

# Basic statistics
print(f"\nTotal reviews: {len(df)}")
print(f"Predicted positive: {df['pred_label'].sum()} ({100*df['pred_label'].mean():.1f}%)")
print(f"Predicted negative: {len(df) - df['pred_label'].sum()} ({100*(1-df['pred_label'].mean()):.1f}%)")

# Check if 'recommended' column exists
if 'recommended' in df.columns:
    # Clean the recommended column (might be 'yes'/'no' or 1/0)
    df['recommended_binary'] = df['recommended'].map({
        'yes': 1, 'no': 0, 
        1: 1, 0: 0,
        'Yes': 1, 'No': 0
    })
    
    # Remove rows where recommended is missing
    df_clean = df.dropna(subset=['recommended_binary'])
    
    print(f"\nReviews with recommendation data: {len(df_clean)}")
    print(f"Recommended: {df_clean['recommended_binary'].sum()} ({100*df_clean['recommended_binary'].mean():.1f}%)")
    print(f"Not recommended: {len(df_clean) - df_clean['recommended_binary'].sum()} ({100*(1-df_clean['recommended_binary'].mean()):.1f}%)")
    
    # Calculate accuracy
    accuracy = accuracy_score(df_clean['recommended_binary'], df_clean['pred_label'])
    print(f"\n{'='*60}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"{'='*60}")
    
    # Confusion Matrix
    cm = confusion_matrix(df_clean['recommended_binary'], df_clean['pred_label'])
    print("\nConfusion Matrix:")
    print("                 Predicted Negative  Predicted Positive")
    print(f"Actually Negative        {cm[0,0]:6d}              {cm[0,1]:6d}")
    print(f"Actually Positive        {cm[1,0]:6d}              {cm[1,1]:6d}")
    
    # Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(
        df_clean['recommended_binary'], 
        df_clean['pred_label'],
        target_names=['Not Recommended', 'Recommended']
    ))
    
    # Agreement analysis
    agreement = (df_clean['recommended_binary'] == df_clean['pred_label']).sum()
    print(f"\nAgreement: {agreement}/{len(df_clean)} ({100*agreement/len(df_clean):.1f}%)")
    print(f"Disagreement: {len(df_clean)-agreement}/{len(df_clean)} ({100*(len(df_clean)-agreement)/len(df_clean):.1f}%)")
    
    # Analyze disagreements
    disagreements = df_clean[df_clean['recommended_binary'] != df_clean['pred_label']]
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
        print(f"Average confidence for false negatives: {false_negatives['pred_prob_pos'].mean():.3f}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actually Not Recommended', 'Actually Recommended'])
    plt.title('Confusion Matrix: Sentiment vs Recommendation')
    plt.ylabel('Actual (Recommended)')
    plt.xlabel('Predicted (Sentiment)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Confidence distribution
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
    
    # By actual recommendation
    axes[1].hist([df_clean[df_clean['recommended_binary']==0]['pred_prob_pos'],
                  df_clean[df_clean['recommended_binary']==1]['pred_prob_pos']], 
                 bins=20, label=['Not Recommended', 'Recommended'], alpha=0.7, color=['red', 'green'])
    axes[1].set_xlabel('Confidence Score (prob of positive)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution by Actual Recommendation')
    axes[1].legend()
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('confidence_distributions.png', dpi=300, bbox_inches='tight')
    print("Confidence distributions saved as 'confidence_distributions.png'")
    
    # Sample some disagreements
    if len(disagreements) > 0:
        print(f"\n{'='*60}")
        print("SAMPLE DISAGREEMENTS (first 5)")
        print(f"{'='*60}")
        for idx, row in disagreements.head(5).iterrows():
            print(f"\n--- Review {idx} ---")
            print(f"Text: {row['text'][:200]}...")
            print(f"Recommended: {'Yes' if row['recommended_binary']==1 else 'No'}")
            print(f"Predicted: {'Positive' if row['pred_label']==1 else 'Negative'} (confidence: {row['pred_prob_pos']:.3f})")
            if 'overall_rating' in df.columns:
                print(f"Overall Rating: {row['overall_rating']}")

else:
    print("\n⚠️  'recommended' column not found in the dataset!")
    print("Available columns:", df.columns.tolist())

# Additional analysis with overall_rating if available
if 'overall_rating' in df.columns:
    print(f"\n{'='*60}")
    print("SENTIMENT vs OVERALL RATING")
    print(f"{'='*60}")
    
    df_rating = df.dropna(subset=['overall_rating'])
    
    # Average rating by sentiment
    print(f"\nAverage overall rating for predicted negative: {df_rating[df_rating['pred_label']==0]['overall_rating'].mean():.2f}")
    print(f"Average overall rating for predicted positive: {df_rating[df_rating['pred_label']==1]['overall_rating'].mean():.2f}")
    
    # Correlation
    correlation = df_rating['pred_prob_pos'].corr(df_rating['overall_rating'])
    print(f"\nCorrelation between confidence score and overall rating: {correlation:.3f}")

print(f"\n{'='*60}")
print("ASSESSMENT COMPLETE")
print(f"{'='*60}")