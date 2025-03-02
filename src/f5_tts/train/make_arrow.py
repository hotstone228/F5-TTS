from finetune_gradio import *
import tqdm
from datasets import Dataset  # using Hugging Face datasets library

PROHIBITED_SYMBOLS = set(
    [
        '"',
        "#",
        "$",
        "%",
        "&",
        "*",
        "++",
        "+,",
        "+.",
        "/",
        ":",
        ";",
        "=",
        "[",
        "]",
        "_",
        "~",
        "¡",
        "£",
        "§",
        "«",
        "°",
        "±",
        "´",
        "·",
        "»",
        "ß",
        "à",
        "á",
        "â",
        "ã",
        "ä",
        "å",
        "æ",
        "ç",
        "è",
        "é",
        "ê",
        "ë",
        "ì",
        "í",
        "î",
        "ï",
        "ð",
        "ñ",
        "ò",
        "ó",
        "ô",
        "õ",
        "ö",
        "ø",
        "ù",
        "ú",
        "û",
        "ü",
        "ý",
        "ā",
        "ă",
        "ć",
        "č",
        "ē",
        "ę",
        "ě",
        "ğ",
        "ī",
        "ı",
        "ł",
        "ń",
        "ň",
        "ō",
        "ő",
        "œ",
        "ř",
        "ş",
        "š",
        "ū",
        "ž",
        "ǎ",
        "ǐ",
        "ǒ",
        "ș",
        "ə",
        "ʻ",
        "α",
        "β",
        "κ",
        "π",
        "χ",
        "ب",
        "ت",
        "ح",
        "د",
        "ص",
        "ع",
        "ل",
        "م",
        "ن",
        "ه",
        "ي",
        "ṃ",
        "ạ",
        "ả",
        "ị",
        "ụ",
        "‑",
        "–",
        "—",
        "…",
        "€",
        "→",
        "≡",
        "、",
        "。",
        "ﷺ",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
)


def create_metadata(name_project, ch_tokenizer):
    # Define file paths
    path_project = os.path.join(path_data, name_project)
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")
    file_raw = os.path.join(path_project, "raw.csv")
    file_duration = os.path.join(path_project, "duration1.json")
    file_vocab = os.path.join(path_project, "vocab.txt")
    file_allowed = os.path.join(path_project, "vocab.txt")

    with open(file_allowed, encoding="utf-8") as f:
        allowed = f.read().split("\n")

    # Check if metadata file exists
    if not os.path.isfile(file_metadata):
        return "The file was not found in " + file_metadata, ""

    batch_size = 10000  # (kept for reference, though not used for incremental writes)
    records = []  # List to accumulate all records
    duration_list = []  # Collect durations (manageable in memory)
    text_vocab_set = set()  # Collect unique tokens for vocabulary
    total_duration = 0  # Track total duration incrementally

    # Process metadata file line by line
    with open(file_metadata, "r", encoding="utf-8") as data:
        for line in tqdm.tqdm(data, desc="Processing metadata", total=8000000):
            sp_line = line.strip().split("|")
            if len(sp_line) != 2:
                continue
            name_audio, text = sp_line[0], sp_line[1]

            if len(text) < 3:
                continue

            # Process text
            text = clear_text(text)
            text = re.findall("\+.|.", text)

            if any(i in text for i in PROHIBITED_SYMBOLS):
                continue
            if ".opus" in name_audio:
                continue

            # Get and validate audio file path
            file_audio = get_correct_audio_path(name_audio, path_project_wavs)
            if not os.path.isfile(file_audio):
                continue

            # Get audio duration (dummy value used here)
            try:
                # duration = get_audio_duration(file_audio)
                duration = 5
            except Exception as e:
                print(f"Error processing {file_audio}: {e}")
                continue

            # Filter by duration and text length
            if duration < 1 or duration > 25:
                continue

            # Create record and add to list
            record = {"audio_path": file_audio, "text": text, "duration": duration}
            records.append(record)
            duration_list.append(duration)
            total_duration += duration
            if ch_tokenizer:
                text_vocab_set.update(text)

    # Check if any valid data was processed
    if not duration_list:
        return (
            f"Error: No audio files found in the specified path: {path_project_wavs}",
            "",
        )

    # Create dataset from the list of records and save as a parquet file
    dataset = Dataset.from_list(records)
    dataset.to_csv(file_raw)

    # Calculate duration statistics
    min_second = round(min(duration_list), 2)
    max_second = round(max(duration_list), 2)

    # Write durations to JSON file
    with open(file_duration, "w") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Handle vocabulary file
    new_vocal = ""
    if not ch_tokenizer:
        if not os.path.isfile(file_vocab):
            file_vocab_finetune = os.path.join(
                path_data, "Emilia_ZH_EN_pinyin/vocab.txt"
            )
            if not os.path.isfile(file_vocab_finetune):
                return "Error: Vocabulary file 'Emilia_ZH_EN_pinyin' not found!", ""
            shutil.copy2(file_vocab_finetune, file_vocab)

        with open(file_vocab, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char.strip()] = i
        vocab_size = len(vocab_char_map)
    else:
        with open(file_vocab, "w", encoding="utf-8") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")
                new_vocal += vocab + "\n"
        vocab_size = len(text_vocab_set)

    return (
        f"prepare complete \nsamples: {len(duration_list)}\ntime data: {format_seconds_to_hms(total_duration)}\nmin sec: {min_second}\nmax sec: {max_second}\nfile_parquet: {file_raw}\nvocab: {vocab_size}",
        new_vocal,
    )


print(create_metadata("russian_custom", True))
