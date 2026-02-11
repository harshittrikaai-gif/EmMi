#!/usr/bin/env python3
"""
Download pre-cleaned datasets from Hugging Face.
FASTEST option - no processing needed!
"""

import os
from tqdm import tqdm

# Pre-cleaned, high-quality datasets
DATASETS = {
    'en': {
        'name': 'wikipedia',
        'config': '20220301.en',
        'split': 'train[:100000]',  # First 100K articles
        'text_column': 'text'
    },
    'hi': {
        'name': 'ai4bharat/IndicCorp',
        'config': 'hi',
        'split': 'train[:100000]',
        'text_column': 'text'
    },
    'bn': {
        'name': 'ai4bharat/IndicCorp',
        'config': 'bn',
        'split': 'train[:50000]',
        'text_column': 'text'
    },
    'ta': {
        'name': 'ai4bharat/IndicCorp',
        'config': 'ta',
        'split': 'train[:50000]',
        'text_column': 'text'
    },
    'te': {
        'name': 'ai4bharat/IndicCorp',
        'config': 'te',
        'split': 'train[:50000]',
        'text_column': 'text'
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
    
    print(f"\n📥 Downloading {lang.upper()} dataset...")
    print(f"   Source: {config['name']}")
    
    try:
        # Load dataset
        dataset = load_dataset(
            config['name'],
            config.get('config'),
            split=config['split'],
            trust_remote_code=True
        )
        
        # Extract text
        output_file = os.path.join(output_dir, f'{lang}.txt')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"   Processing {len(dataset):,} examples...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Writing {lang}"):
                text = example[config['text_column']]
                
                # Split into sentences
                sentences = text.split('\n')
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 20:  # Filter short sentences
                        f.write(sent + '\n')
        
        file_size = os.path.getsize(output_file) / 1024 / 1024
        print(f"✅ Saved to {output_file}")
        print(f"   File size: {file_size:.1f} MB")
        
        # Count lines
        with open(output_file, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        print(f"   Total sentences: {num_lines:,}")
        
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
    
    print("🚀 Emmit AI - Fast Dataset Downloader")
    print("=" * 60)
    print("\n⚠️  Make sure you have 'datasets' installed:")
    print("   pip install datasets")
    print("\n")
    
    for lang in args.languages:
        download_dataset(lang, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print(f"\n📁 Your data is in: {args.output_dir}/")
    print("\n🎯 Next step:")
    print("   python scripts/train_tokenizer.py --data_path data/raw/<lang>.txt --vocab_size 32000 --output tokenizers/my_tokenizer")

if __name__ == '__main__':
    main()
