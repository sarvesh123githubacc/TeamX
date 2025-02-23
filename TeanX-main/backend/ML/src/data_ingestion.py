import pandas as pd

def load_and_merge_datasets(file1, file2, save_path):
    """
    Loads two datasets, merges them on 'text' and 'label', and saves the combined dataset.
    
    Parameters:
    file1 (str): Path to the first dataset.
    file2 (str): Path to the second dataset.
    save_path (str): Path to save the combined dataset.
    """
    try:
        # Load datasets
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Concatenate both datasets
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save the combined dataset
        combined_df.to_csv(save_path, index=False)
        print(f"✅ Combined dataset saved successfully at: {save_path}")
        
        return combined_df
    
    except Exception as e:
        print(f"❌ Error in data ingestion: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file1 = "data/news_dataset1.csv"  # Update with actual path
    file2 = "data/news_dataset2.csv"  # Update with actual path
    save_path = "data/combined_news.csv"
    
    combined_df = load_and_merge_datasets(file1, file2, save_path)