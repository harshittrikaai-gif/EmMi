#!/usr/bin/env python3
"""
Download pre-cleaned datasets from Hugging Face.
FASTEST option - no processing needed!
"""

import os
from tqdm import tqdm

# Pre-cleaned, high-quality datasets (using modern HF formats)
# Pre-cleaned, high-quality datasets
DATASETS = {
    'en': {
        'name': 'wikitext',
        'config': 'wikitext-103-v1',
        'split': 'train',
        'text_column': 'text',
        'limit': 5000
    },
    'hi': {
        'name': 'vukuzmanovic/hindi-wikipedia-clean',
        'config': None,
        'split': 'train',
        'text_column': 'text',
        'limit': 5000
    },
    'bn': {
        'name': 'ai4bharat/IndicCorp',
        'config': 'bn',
        'split': 'train',
        'text_column': 'text',
        'limit': 5000
    }
}

def download_dataset(lang, output_dir='data/raw'):
    """Download dataset from Hugging Face."""
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not found. Run: pip install datasets")
        return None

    if lang not in DATASETS:
        print(f"❌ Language {lang} not configured")
        return None
    
    config = DATASETS[lang]
    
    print(f"\n[INFO] Downloading {lang.upper()} dataset...")
    print(f"   Source: {config['name']}")
    
    try:
        # Load dataset with streaming to save memory
        # Note: trust_remote_code=True is needed for some configs
        dataset = load_dataset(
            config['name'],
            config.get('config'),
            split=config['split'],
            streaming=True,
            trust_remote_code=True
        )
        
        # Extract text
        output_file = os.path.join(output_dir, f'{lang}.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        limit = config.get('limit', 5000)
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Writing {lang}"):
                text = example[config['text_column']]
                
                # Split into sentences
                sentences = text.split('\n')
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 40:  # Higher threshold for better quality
                        f.write(sent + '\n')
                
                count += 1
                if count >= limit:
                    break
        
        file_size = os.path.getsize(output_file) / 1024 / 1024
        print(f"✅ Saved {count:,} examples to {output_file}")
        print(f"   File size: {file_size:.1f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error downloading {lang}: {e}")
        return None

def main():
    """Download all configured datasets."""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', 
                       default=['en', 'hi', 'bn'],
                       help='Languages to download')
    parser.add_argument('--output_dir', default='data/raw',
                       help='Output directory')
    args = parser.parse_args()
    
    print("Emmit AI - Fast Dataset Downloader")
    print("=" * 60)
    print("\n[NOTE] Make sure you have 'datasets' installed:")
    print("   pip install datasets")
    print("\n")
    
    for lang in args.languages:
        download_dataset(lang, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"\n[PATH] Your data is in: {args.output_dir}/")
    print("\n[NEXT STEP]")
    print("   python scripts/train_tokenizer.py --data_path data/raw/<lang>.txt --vocab_size 32000 --output tokenizers/my_tokenizer")

if __name__ == '__main__':
    main()
