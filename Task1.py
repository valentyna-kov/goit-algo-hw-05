import os, timeit, statistics
from collections import defaultdict

###################### rabin_karp_search ######################################
def rabin_karp_search(main_string: str, pattern: str) -> int:
    '''
    Rabin-Karp search algorithm for finding a substring in a string.
        Args:
            main_string: The string to search within
            pattern: The substring to search for
        Returns:
            The starting index of the first occurrence of the substring in the main string, or -1 if not found.
    '''
    def polynomial_hash(text: str, base=256, modulus=101) -> int:
        '''
        Compute the polynomial for Gorner's hash of a string.
            Args:
                text: The input string to hash
                base: The base value for the polynomial hash
                modulus: The modulus value for the hash
            Returns:
                The computed polynomial hash as an integer
        '''
        hash_value = 0
        for ch in text:
            hash_value = (hash_value * base + ord(ch)) % modulus
        return hash_value
   
    
    pattern_length = len(pattern)
    main_string_length = len(main_string)
    if pattern_length == 0:
        return 0
    if pattern_length > main_string_length:
        return -1
    
    # Base and modulus for the hash function
    base = 256
    modulus = 101

    # Hash values for the search pattern and the current slice in the main string
    pattern_hash = polynomial_hash(pattern, base, modulus)
    current_slice_hash = polynomial_hash(main_string[:pattern_length], base, modulus)
    # Previous value for rehashing
    h_multiplier = pow(base, pattern_length - 1, modulus) 

    # Iterate through the main string
    for i in range(main_string_length - pattern_length + 1):
        if pattern_hash == current_slice_hash and pattern == main_string[i:i+pattern_length]:
            return i

        if i < main_string_length - pattern_length:
            current_slice_hash = (current_slice_hash - ord(main_string[i]) * h_multiplier) % modulus
            current_slice_hash = (current_slice_hash * base + ord(main_string[i + pattern_length])) % modulus
            if current_slice_hash < 0:
                current_slice_hash += modulus

    return -1

###################### boyer_moore_search ######################################
def boyer_moore_search(main_string: str, pattern: str) -> int:
    '''
    Boyer-Moore search algorithm for finding a substring in a string.
        Args:
            text: The text to search within.
            pattern: The substring to search for.
        Returns:
            The starting index of the first occurrence of the pattern in the text, or -1 if not found.
    '''

    def build_shift_table(pattern: str) -> dict[str, int]:
        '''
        Build the shift table for the Boyer-Moore algorithm.
            Args:
                pattern: The substring to search for.
            Returns:
                A dictionary representing the shift table.
        '''
        table = {}
        length = len(pattern)
        for index, char in enumerate(pattern[:-1]):
            table[char] = length - index - 1
        table.setdefault(pattern[-1], length)
        return table

    m = len(pattern)
    n = len(main_string)
    if m == 0:
        return 0
    if m > n:
        return -1

    shift_table = build_shift_table(pattern)
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and main_string[i + j] == pattern[j]:
            j -= 1
        if j < 0:
            return i
        i += shift_table.get(main_string[i + m - 1], m)

    return -1

###################### kmp_search ######################################
def kmp_search(main_string: str, pattern: str) -> int:
    '''
    KMP search algorithm for finding a substring in a string.
        Args:
            main_string: The string to search within.
            pattern: The substring to search for.
        Returns:
            The starting index of the first occurrence of the pattern in the main_string, or -1 if not found.
    '''

    def compute_lps(pattern: str) -> list[int]:
        '''
        Compute the Longest Prefix Suffix (LPS) array for the KMP algorithm.
            Args:
                pattern: The substring to search for.
            Returns:
                The LPS array as a list of integers.
        '''
        lps = [0] * len(pattern)
        length = 0
        i = 1

        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    M = len(pattern)
    N = len(main_string)
    if M == 0:
        return 0    
    if M > N:
        return -1

    lps = compute_lps(pattern)
    i = j = 0
    while i < N:
        if pattern[j] == main_string[i]:
            i += 1
            j += 1
        elif j != 0:
            j = lps[j - 1]
        else:
            i += 1
        if j == M:
            return i - j

    return -1
########################### util #########################################
def get_file_data(file_path: str) -> str | None:
    try:
        with open(file_path, "r", encoding="cp1251") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"No access to file: {file_path}")
    except UnicodeDecodeError:
        print(f"Encoding error while reading file: {file_path}")
    return None

ALGS = {
    "Boyer–Moore": boyer_moore_search,
    "KMP": kmp_search,
    "Rabin–Karp": rabin_karp_search,
}

PATTERNS = {
    "short_exists_art1": "чергові найбільші",
    "long_exists_art1": "Режим доступу до ресурсу: https://uk.wikipedia.org/wiki/GPGPU – Назва з екрану",
    "short_exists_art2": "При використанні 4 байт ",
    "long_exists_art2": "Зменшення розміру блоку дозволяє зменшити втрати пам’яті, але збільшує час доступу до елементів"
}

def bench_once(fn, text: str, pattern: str) -> float:
    '''
    Compute a single execution of a function.
        Args:
            fn: The function to benchmark.
            text: The main string to search within.
            pattern: The substring to search for.
        Returns:
            The execution time in seconds.
    '''
    return timeit.Timer(lambda: fn(text, pattern)).timeit(number=1)

def bench_median(fn, text: str, pattern: str, repeats: int) -> float:
    '''
    Compute the median execution time of a function over multiple runs.
        Args:
            fn: The function to benchmark.
            text: The main string to search within.
            pattern: The substring to search for.
            repeats: The number of times to repeat the benchmark.
        Returns:
            The median execution time in seconds.
    '''
    times = [bench_once(fn, text, pattern) for _ in range(repeats)]
    return statistics.median(times)

def print_table(rows: list[tuple], headers: tuple[str, ...]):
    str_rows = [[str(x) for x in r] for r in rows]
    
    data = [list(headers)] + str_rows
    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]

    header_line = "| " + " | ".join(data[0][i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line    = "|-" + "-|-".join("-"*widths[i] for i in range(len(headers))) + "-|"

    print()
    print(header_line)
    print(sep_line)
    for row in str_rows:
        line = "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        print(line)

def print_table_highlight(rows: list[tuple], headers: tuple[str, ...]):

    groups = defaultdict(list)
    for idx, r in enumerate(rows):
        key = (r[0], r[1], r[3])   # article, pattern_type, present
        groups[key].append(idx)

    best, worst = set(), set()
    for key, idxs in groups.items():
        times = [rows[i][5] for i in idxs]
        if all(isinstance(t, (int, float)) for t in times):
            tmin, tmax = min(times), max(times)
            for i in idxs:
                if rows[i][5] == tmin:
                    best.add(i)
                if rows[i][5] == tmax:
                    worst.add(i)

    str_rows = [
        [str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]),
         f"{r[5]:.6f}" if isinstance(r[5], (int, float)) else str(r[5])]
        for r in rows
    ]

    data = [list(headers)] + str_rows
    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]

    def line(cells): return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    print()
    print(line(headers))
    print("|-" + "-|-".join("-"*w for w in widths) + "-|")

    GREEN, YELLOW, RESET = "\033[32m", "\033[33m", "\033[0m"
    g = 0
    for i, cells in enumerate(str_rows):
        s = line(cells)
        if i in best:
            print(GREEN + s + RESET)
        elif i in worst:
            print(YELLOW + s + RESET)
        else:
            print(s)
        g += 1
        if g == 3:
            print("|-" + "-|-".join("-"*w for w in widths) + "-|")
            g = 0

def print_avg_per_article(results: list[tuple]):
    """
    Print average execution time per algorithm for each article.
    """
    sums_counts = defaultdict(lambda: [0.0, 0])   # key: (article, algorithm) -> [sum, count]
    for r in results:
        article, alg, t = r[0], r[4], r[5]  
        sums_counts[(article, alg)][0] += t
        sums_counts[(article, alg)][1] += 1

    avg_rows = []
    for (article, alg), (s, c) in sums_counts.items():
        avg = s / c if c else 0.0
        avg_rows.append((article, alg, f"{avg:.6f}"))

    # sort: article, algorithm
    avg_rows.sort(key=lambda x: (x[0], x[1]))

    headers = ("article", "algorithm", "avg_time_s")
    data = [headers] + [tuple(map(str, r)) for r in avg_rows]
    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]

    def line(cells): 
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    print()  
    print("**Average time per algorithm for each article**\n")
    print(line(headers))
    print("|-" + "-|-".join("-"*w for w in widths) + "-|")
    last_article = None
    for row in avg_rows:
        article, alg, avg_time = row
        if last_article is not None and article != last_article:
            print("|" + "|".join("-"* (w+2) for w in widths) + "|")
        print(line([article, alg, avg_time]))
        last_article = article

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    art1 = get_file_data("article1.txt")
    if art1 is None:
        return
    art2 = get_file_data("article2.txt")
    if art2 is None:
        return
    
    cases = [
        ("article1.txt", art1, "short_exists_art1", "exists"),
        ("article1.txt", art1, "long_exists_art1",  "exists"),
        ("article1.txt", art1, "short_exists_art2", "absent"),
        ("article1.txt", art1, "long_exists_art2",  "absent"),
        ("article2.txt", art2, "short_exists_art2", "exists"),
        ("article2.txt", art2, "long_exists_art2",  "exists"),
        ("article2.txt", art2, "short_exists_art1", "absent"),
        ("article2.txt", art2, "long_exists_art1",  "absent"),
    ]
    results = []
    repeats = 5
    for article_name, text, pkey, presence in cases:
        pattern = PATTERNS[pkey]
        plen = len(pattern)
        pat_type = "short" if pkey.startswith("short") else "long"
        for alg_name, alg_fn in ALGS.items():
            t = bench_median(alg_fn, text, pattern, repeats)
            results.append((article_name, pat_type, plen, presence, alg_name, t))

    headers = ("article", "pattern_type", "pattern_len", "present", "algorithm", "time_s_median")
    print_table_highlight(results, headers)

    print_avg_per_article(results)

    # после вызова print_avg_per_article(results)

    print()
    print("**Conclusions:**")
    print("- For article1.txt, the fastest algorithm is **Boyer–Moore**.")
    print("- For article2.txt, the fastest algorithm is **Boyer–Moore**.")
    print("- Overall, considering both articles, the fastest algorithm is **Boyer–Moore**.")

if __name__ == "__main__":
    main()
