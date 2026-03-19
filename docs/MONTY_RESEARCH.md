# Monty Research Notes

> Bu doküman, `pydantic-monty` projesinin evalspec ekosistemi (vowel eval generation pipeline'ları ve vowel-optimization) ile entegrasyonu için yapılan araştırmanın özetidir. CodeMode, tüm eval generation pipeline'larında kullanılabilecek genel bir mekanizmadır — optimizasyon bunlardan sadece biridir.

## 1. Genel Bakış

**Monty**, Pydantic ekibi tarafından Rust ile yazılmış, minimal ve güvenli bir Python yorumlayıcısıdır. Temel amacı: **AI tarafından üretilen kodu güvenli bir sandbox ortamında çalıştırmak.**

- **Repo:** `pydantic/monty` (GitHub)
- **PyPI paketi:** `pydantic-monty`
- **NPM paketi:** `@pydantic/monty`
- **Lisans:** MIT
- **Dil:** Rust (PyO3 ile Python bindings, napi-rs ile JS bindings)
- **Hedef Python sürümü:** 3.14

### Temel Özellikler

| Özellik | Detay |
|---------|-------|
| Güvenlik | Filesystem, network, env vars tamamen bloklu — sadece kontrollü external function callbacks |
| Başlatma süresi | <0.06ms (~60 mikrosaniye) |
| Performans | CPython'a benzer çalışma hızı |
| Boyut | ~4.5MB download |
| Serileştirme | `dump()`/`load()` ile parsed code ve execution state kaydedilebilir |
| Kaynak limitleri | Süre, bellek, allocation sayısı, recursion derinliği sınırlandırılabilir |
| Tip kontrolü | Opsiyonel statik tip analizi (Monty'nin kendi type checker'ı) |

## 2. Güvenlik Modeli

Monty, **untrusted/potentially malicious** kodun çalıştırılması için tasarlanmıştır. Güvenlik garantileri:

- **Filesystem erişimi YOK** — Sadece `OSAccess` ile kontrollü sanal dosya sistemi
- **Network erişimi YOK** — Socket, HTTP vb. hiçbir ağ işlemi yapılamaz
- **Ortam değişkenleri YOK** — `os.environ`, `os.getenv` yalnızca host callback ile
- **Subprocess/shell YOK** — `os.system`, `subprocess` vb. yok
- **Import sistemi kısıtlı** — Sadece izin verilen modüller (sys, typing, asyncio)
- **C FFI yok** — Tamamen Rust ile implement edilmiş, unsafe yok

Tüm dış dünya erişimi **external functions** mekanizması üzerinden olur — host tarafı bu fonksiyonları sağlar, sandbox kodu bunları çağırır, host gerçek işlemi yapar ve sonucu sandbox'a döndürür.

## 3. Python API

### 3.1. Kurulum

```bash
pip install pydantic-monty
```

### 3.2. Temel Kullanım

```python
import pydantic_monty

# Basit ifade çalıştırma
m = pydantic_monty.Monty('1 + 2 * 3')
result = m.run()  # -> 7

# Input değişkenleri ile
m = pydantic_monty.Monty('x + y', inputs=['x', 'y'])
result = m.run(inputs={"x": 10, "y": 20})  # -> 30

# Aynı parsed code farklı girdilerle tekrar çalıştırılabilir
result2 = m.run(inputs={"x": 100, "y": 200})  # -> 300
```

### 3.3. `Monty` Sınıfı — Constructor

```python
pydantic_monty.Monty(
    code: str,                                    # Çalıştırılacak Python kodu
    *,
    script_name: str = 'main.py',                 # Traceback'lerde görünecek isim
    inputs: list[str] | None = None,              # Kod içinde kullanılabilecek input değişken isimleri
    external_functions: list[str] | None = None,   # Kod içinden çağrılabilecek harici fonksiyon isimleri
    type_check: bool = False,                      # Statik tip kontrolü yapılsın mı
    type_check_stubs: str | None = None,           # Tip kontrolü için ek stub tanımları
    dataclass_registry: list[type] | None = None,  # Dataclass tip kayıtları
)
```

**Raises:**
- `MontySyntaxError` — Kod parse edilemezse
- `MontyTypingError` — `type_check=True` ise ve tip hataları varsa

### 3.4. `Monty.run()` — Senkron Çalıştırma

```python
m.run(
    *,
    inputs: dict[str, Any] | None = None,                          # Input değerleri
    limits: ResourceLimits | None = None,                           # Kaynak limitleri
    external_functions: dict[str, Callable[..., Any]] | None = None, # Harici fonksiyon implementasyonları
    print_callback: Callable[[Literal['stdout'], str], None] | None = None,  # print() çıktısı callback
    os: Callable[[OsFunction, tuple[Any, ...]], Any] | None = None,          # OS erişimi callback
) -> Any
```

**Önemli:** GIL serbest bırakılır — paralel çalıştırma mümkün.

### 3.5. External Functions (Harici Fonksiyonlar)

Bu, Monty'nin en güçlü mekanizmasıdır. Sandbox kodu bir fonksiyon çağırdığında, çalışma durur, host taraftaki gerçek Python fonksiyonu çalışır ve sonuç sandbox'a döndürülür.

```python
# Sandbox kodunda "fetch" fonksiyonu çağrılabilir
m = pydantic_monty.Monty(
    'fetch("https://example.com")',
    external_functions=['fetch']
)

# Host tarafında gerçek implementasyon
def fetch(url: str) -> str:
    return f'Fetched: {url}'

result = m.run(external_functions={"fetch": fetch})
# -> "Fetched: https://example.com"
```

**Kritik nokta:** External fonksiyonlar host ortamında çalışır — yani hedef fonksiyonun stdlib, third-party lib, dosya sistemi vb. kullanması sorun olmaz. Monty sadece orkestrasyonu yapar.

### 3.6. İteratif Çalıştırma (start/resume)

External fonksiyon çağrılarında adım adım kontrol sağlar:

```python
m = pydantic_monty.Monty(
    'result = fetch(url)',
    inputs=['url'],
    external_functions=['fetch']
)

# Çalıştırmayı başlat
progress = m.start(inputs={"url": "https://example.com"})

if isinstance(progress, pydantic_monty.MontySnapshot):
    # Bir external function çağrısında durdu
    print(progress.function_name)   # -> "fetch"
    print(progress.args)            # -> ("https://example.com",)
    print(progress.kwargs)          # -> {}
    
    # Sonucu döndürerek devam et
    progress = progress.resume(return_value="response data")

if isinstance(progress, pydantic_monty.MontyComplete):
    print(progress.output)  # -> Son ifadenin değeri
```

**İlerleme tipleri:**
- `MontySnapshot` — External function çağrısı bekliyor
- `MontyFutureSnapshot` — Birden fazla async future bekliyor
- `MontyComplete` — Çalışma tamamlandı, `.output` ile sonuç alınır

### 3.7. Asenkron Çalıştırma

```python
async def main():
    m = pydantic_monty.Monty(
        'await fetch(url)',
        inputs=['url'],
        external_functions=['fetch']
    )
    
    async def real_fetch(url: str) -> str:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            return r.text
    
    result = await pydantic_monty.run_monty_async(
        m,
        inputs={"url": "https://example.com"},
        external_functions={"fetch": real_fetch},
    )
```

### 3.8. REPL Modu

Durum korunarak ardışık kod parçaları çalıştırılabilir:

```python
repl, output = pydantic_monty.MontyRepl.create('x = 10', inputs=['x'])
# output = 10 (veya None — son ifadenin değeri)

result1 = repl.feed('x + 5')    # -> 15
result2 = repl.feed('x * 2')    # -> 20
# x hâlâ 10, önceki state korunur
```

### 3.9. Kaynak Limitleri (ResourceLimits)

```python
limits = pydantic_monty.ResourceLimits(
    max_duration_secs=5.0,        # Maksimum çalışma süresi (saniye)
    max_memory=1024 * 1024,        # Maksimum heap bellek (byte) — 1MB
    max_allocations=10000,         # Maksimum heap allocation sayısı
    max_recursion_depth=1000,      # Maksimum recursion derinliği (default: 1000)
    gc_interval=100,               # Her N allocation'da GC çalıştır
)

m = pydantic_monty.Monty('fib(30)', external_functions=['fib'])
result = m.run(
    external_functions={"fib": my_fib},
    limits=limits,
)
```

### 3.10. Serileştirme (dump/load)

Parsed code veya çalışma durumu (snapshot) kaydedilebilir:

```python
# Parsed code'u kaydet
m = pydantic_monty.Monty('x + 1', inputs=['x'])
data = m.dump()  # -> bytes

# Daha sonra geri yükle (parse maliyeti sıfır)
m2 = pydantic_monty.Monty.load(data)
result = m2.run(inputs={"x": 41})  # -> 42

# Snapshot'ü da kaydedebilirsin
progress = m.start(inputs={"x": 10})
if isinstance(progress, pydantic_monty.MontySnapshot):
    snapshot_data = progress.dump()  # -> bytes
    # Farklı process'te bile geri yüklenebilir
    restored = pydantic_monty.MontySnapshot.load(snapshot_data)
```

### 3.11. Sanal Dosya Sistemi (OSAccess)

```python
from pydantic_monty import OSAccess, MemoryFile, CallbackFile

# Bellekte sanal dosyalar oluştur
fs = OSAccess([
    MemoryFile('/data/input.csv', content='col1,col2\n1,2\n3,4'),
    MemoryFile('/data/config.json', content='{"key": "value"}'),
])

# Sandbox kodunda Path.read_text() vb. kullanılabilir
m = pydantic_monty.Monty("""
from pathlib import Path
data = Path('/data/input.csv').read_text()
data.split('\\n')
""")

result = await pydantic_monty.run_monty_async(m, os=fs)
```

### 3.12. Tip Kontrolü

```python
# Opsiyonel statik analiz
m = pydantic_monty.Monty(
    'x + "hello"',
    inputs=['x'],
    type_check=True,
    type_check_stubs='x: int',  # Input tiplerini belirt
)
# MontyTypingError fırlatabilir

# Hata formatları
try:
    m.type_check(prefix_code='x: int')
except pydantic_monty.MontyTypingError as e:
    print(e.display(format='full', color=True))
    # format seçenekleri: 'full', 'concise', 'azure', 'json', 'jsonlines',
    #                      'rdjson', 'pylint', 'gitlab', 'github'
```

## 4. Hata Tipleri

```
MontyError (base)
├── MontySyntaxError    — Parse hataları
├── MontyRuntimeError   — Çalışma zamanı hataları (ZeroDivisionError, ValueError vb.)
└── MontyTypingError    — Statik tip analizi hataları
```

### MontyRuntimeError Detayları

```python
try:
    m = pydantic_monty.Monty('1 / 0')
    m.run()
except pydantic_monty.MontyRuntimeError as e:
    # İç exception'a eriş
    inner = e.exception()  # -> ZeroDivisionError instance
    
    # Traceback al
    frames = e.traceback()  # -> list[Frame]
    for frame in frames:
        print(f"  {frame.filename}:{frame.line}:{frame.column} in {frame.function_name}")
        print(f"    {frame.source_line}")
    
    # Formatlanmış çıktı
    print(e.display(format='traceback'))  # Full traceback
    print(e.display(format='type-msg'))   # "ZeroDivisionError: division by zero"
    print(e.display(format='msg'))        # "division by zero"
```

**ÖNEMLİ:** Monty, Python exception'larını birebir eşleştirir. `ZeroDivisionError`, `ValueError`, `TypeError` vb. host tarafında doğru exception tipleri olarak yakalanabilir.

## 5. Dil Destekleri ve Kısıtlamalar

### 5.1. Desteklenen Python Deyimleri (Statements)

Kaynak: `crates/monty/src/expressions.rs` — `Node` ve `Expr` enum'ları

| Deyim | Notlar |
|-------|--------|
| `x = expr` | Basit atama |
| `x, y = expr` | Tuple unpacking (iç içe dahil: `(a, b), c = ...`) |
| `first, *rest = expr` | Starred unpacking |
| `x += expr` (augmented assigns) | `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `\|=`, `^=`, `<<=`, `>>=` |
| `obj[i] = val` | Subscript assignment |
| `obj.attr = val` | Attribute assignment (dataclass alanları) |
| `if / elif / else` | Tam destekli |
| `for target in iter` | `else` bloğu dahil |
| `while test` | `else` bloğu dahil |
| `break` | ✅ |
| `continue` | ✅ |
| `return` / `return expr` | ✅ |
| `raise` / `raise Exception(...)` | ✅ |
| `try / except / else / finally` | Tam hiyerarşi destekli, çoklu `except` |
| `assert test, msg` | ✅ |
| `pass` | ✅ |
| `def func(...)` | `async def` dahil |
| `global x` | ✅ |
| `nonlocal x` | ✅ |
| `import sys` | Sadece whitelist'teki modüller |
| `from typing import X` | Sadece whitelist'teki modüller |
| `del` | ❌ Henüz yok |
| `class MyClass:` | ❌ Henüz yok |
| `match x:` | ❌ Desteklenmiyor |
| `with ... as ...:` | ❌ Henüz yok |

### 5.2. Desteklenen İfadeler (Expressions)

| İfade | Notlar |
|-------|--------|
| Literaller | `int`, `float`, `str`, `bytes`, `bool`, `None`, `...` |
| Büyük int'ler | `2**200` gibi i64 aşan değerler (arbitrary precision) |
| f-string | `f"hello {name!r}"` — format spec dahil |
| Aritmetik | `+`, `-`, `*`, `/`, `//`, `%`, `**` |
| Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` |
| Karşılaştırma | `==`, `!=`, `<`, `<=`, `>`, `>=`, `is`, `is not`, `in`, `not in` |
| Zincirleme karşılaştırma | `a < b < c` — kısa devre değerlendirmeli |
| Boolean | `and`, `or`, `not` |
| Unary | `-x`, `+x`, `~x` |
| Ternary | `x if cond else y` |
| Walrus | `(x := expr)` |
| `await expr` | Modül seviyesinde de kullanılabilir (Jupyter tarzı) |
| List/dict/set literali | `[1,2]`, `{k:v}`, `{1,2}` |
| List/set/dict comprehension | `[x for x in iter if cond]` |
| Generator expression | `(x for x in iter)` |
| Lambda | `lambda x, y: x + y` |
| Subscript | `obj[i]`, `obj[a:b:c]` |
| Slice | `obj[::2]` |
| Attribute erişimi | `obj.attr` (zincirli dahil) |
| Fonksiyon çağrısı | `f(a, b, *args, key=val, **kwargs)` |
| Method çağrısı | `obj.method(args)` |
| `isinstance(obj, Type)` | ✅ |

### 5.3. Desteklenen Yerleşik Tipler (Built-in Types)

```
bool  int  float  str  bytes
list  tuple  dict  set  frozenset
range  slice  iter
type  property
```

Ayrıca:
- `None`, `True`, `False`, `...` (Ellipsis)
- `LongInt` — arbitrarily large integers
- `NamedTuple` — `collections.namedtuple` benzeri (built-in desteği var)
- `Dataclass` — `@dataclass` decorator'ı ile (host'tan registry ile)
- `pathlib.Path` — `from pathlib import Path` ile

### 5.4. Desteklenen Builtin Fonksiyonlar

Kaynak: `crates/monty/src/builtins/mod.rs` — `BuiltinsFunctions` enum'u

**Mevcut (✅):**
```
abs()       all()       any()       bin()       chr()
divmod()    enumerate() filter()    getattr()   hash()
hex()       id()        isinstance() len()      map()
max()       min()       next()      oct()       ord()
pow()       print()     repr()      reversed()  round()
sorted()    sum()       type()      zip()
```

**Henüz yok / yorum satırı (❌):**
```
aiter()     anext()     ascii()     breakpoint()
callable()  compile()   dir()       eval()
exec()      format()    globals()   hasattr()
help()      input()     issubclass() iter() [kısmen]
locals()    open()      setattr()   staticmethod()
classmethod() super()  vars()      __import__()
```

**Type constructor olarak kullanılabilenler:**
```
bool()  int()  float()  str()  bytes()
list()  tuple()  dict()  set()  frozenset()
range()  slice()  iter()  type()  property()
```

**Exception constructor'ları:**
```
Exception        BaseException    SystemExit       KeyboardInterrupt
ArithmeticError  OverflowError    ZeroDivisionError
LookupError      IndexError       KeyError
RuntimeError     NotImplementedError  RecursionError
AttributeError   FrozenInstanceError
NameError        UnboundLocalError
ValueError       UnicodeDecodeError
ImportError      ModuleNotFoundError
OSError          FileNotFoundError  FileExistsError
IsADirectoryError  NotADirectoryError
AssertionError   MemoryError      StopIteration
SyntaxError      TimeoutError     TypeError
```

### 5.5. Desteklenen Stdlib Modülleri

#### `sys`
```python
import sys
sys.version        # "3.14.0 (Monty)"
sys.version_info   # named tuple: (major=3, minor=14, micro=0, ...)
sys.platform       # "monty"
sys.stdout         # marker (gerçek I/O yok)
sys.stderr         # marker (gerçek I/O yok)
```

#### `typing`
```python
from typing import (
    TYPE_CHECKING,  # her zaman False
    Any, Optional, Union, List, Dict, Tuple, Set,
    FrozenSet, Callable, Type, Sequence, Mapping,
    Iterable, Iterator, Generator, ClassVar,
    Final, Literal, TypeVar, Generic, Protocol,
    Annotated, Self, Never, NoReturn
)
```
Bunlar runtime'da `Marker` değerleri olarak işlenir — tip anotasyonlarda kullanılabilirler.

#### `asyncio`
```python
import asyncio
asyncio.run(coro)        # await coro ile eşdeğer
asyncio.gather(*coros)   # Eşzamanlı birden fazla coroutine çalıştırma
# create_task, sleep, wait vb. → YOK
```

#### `os`
```python
import os
os.getenv("KEY", default=None)  # host callback üzerinden
os.environ                       # host callback üzerinden dict döner
# os.path, os.listdir, os.system vb. → YOK
```

#### `pathlib`
```python
from pathlib import Path
p = Path("/data/file.txt")

# Pure methods (I/O gerektirmez — doğrudan çalışır):
p.name         # "file.txt"
p.stem         # "file"
p.suffix       # ".txt"
p.suffixes     # [".txt"]
p.parent       # Path("/data")
p.parts        # ["/", "data", "file.txt"]
p / "subdir"   # Path birleştirme (/ operatörü)
str(p)         # "/data/file.txt"

# Filesystem methods (OSAccess host callback gerektirir):
p.exists()     read_text()   read_bytes()
p.is_file()    write_text()  write_bytes()
p.is_dir()     mkdir()       unlink()
p.is_symlink() rmdir()       iterdir()
p.stat()       rename()      resolve()
p.absolute()
```

### 5.6. Tip Metodları — Detay

#### `str` metodları
```
capitalize  casefold    center      count       encode
endswith    find        index       isalnum     isalpha
isascii     isdecimal   isdigit     isidentifier islower
isnumeric   isspace     istitle     isupper     join
ljust       lower       lstrip      partition   removeprefix
removesuffix replace    rfind       rindex      rjust
rpartition  rsplit      rstrip      split       splitlines
startswith  strip       swapcase    title       upper      zfill
```
Ayrıca: `+` (concat), `*` (repeat), `in` (contains), `[]` (index/slice), `len()`, `str()` constructor

#### `list` metodları
```
append  clear  copy  count  extend  index  insert  pop  remove  reverse  sort
```
Ayrıca: `+`, `*`, `in`, `[]`, `len()`, comprehension, unpacking

#### `dict` metodları
```
clear  copy  fromkeys  get  items  keys  pop  popitem  setdefault  update  values
```
Ayrıca: `in`, `[]`, `len()`, comprehension

#### `set` / `frozenset` metodları
```
add  clear  copy  difference  discard  intersection  isdisjoint
issubset  issuperset  pop  remove  symmetric_difference  union  update
```
Ayrıca: `|`, `&`, `-`, `^` operatörleri

#### `tuple` metodları
```
count  index
```
Ayrıca: `+`, `*`, `in`, `[]`, `len()`, unpacking

#### `bytes` metodları
```
capitalize  center       count      decode      endswith
find        fromhex      hex        index       isalnum
isalpha     isascii      isdigit    islower     isspace
istitle     isupper      join       ljust       lower
lstrip      partition    removeprefix removesuffix replace
rfind       rindex       rjust      rpartition  rsplit
rstrip      split        splitlines startswith  strip
swapcase    title        upper      zfill
```

#### `int` metodları
```
bit_length  bit_count  to_bytes  from_bytes
```
Ayrıca: tüm aritmetik ve bitwise operatörler

#### `range`
```
range(stop)
range(start, stop)
range(start, stop, step)
```
Iteration, `in`, `len()`, `list(range(...))` desteklenir.

### 5.7. Desteklenmeyen Özellikler

| Özellik | Durum |
|---------|-------|
| **`class` tanımı** | ❌ Henüz yok (geliyor) |
| **`match` / `case`** | ❌ Planlanmamış |
| **`with` / bağlam yöneticisi** | ❌ Henüz yok |
| **`del` deyimi** | ❌ Henüz yok |
| **`yield from`** | ❌ Henüz yok |
| **`*args` spread in comprehension** | ⚠️ Kısıtlı |
| **`eval()`, `exec()`** | ❌ Hiçbir zaman olmayacak |
| **`__import__`** | ❌ Hiçbir zaman olmayacak |
| **Third-party kütüphaneler** | ❌ Sandbox içinde kullanılamaz |
| **`json` modülü** | ❌ Henüz yok (geliyor) |
| **`dataclasses` modülü (import)** | ❌ Henüz yok; dataclass desteği var ama host'tan |
| **`collections`, `itertools`, `math`** | ❌ Yok |
| **`re` (regex)** | ❌ Yok |
| **`datetime`** | ❌ Yok |
| **`functools`** | ❌ Yok |
| **`enum`** | ❌ Yok |
| **Decorator'lar** | ⚠️ Sadece basit fonksiyon decorator'ları |
| **`super()`** | ❌ Yok |
| **`classmethod`, `staticmethod`** | ❌ Yok |

## 6. Mimari (Dahili)

- **Parser:** Ruff'un `ruff_python_parser`'ı kullanılır → AST üretilir
- **Prepare phase:** AST'den Scope analizi yapılır, isimler namespace index'lerine çözümlenir
- **Bytecode:** Hazırlanan AST doğrudan bytecode VM'e beslenir (CPython benzeri register VM)
- **Bellek:** Manuel reference counting (`drop_with_heap`, `clone_with_heap`); GC configurable intervals ile
- **Serileştirme:** `serde` ile binary format (parsed code + snapshot)

### Crate yapısı

| Crate | İçerik |
|-------|--------|
| `crates/monty/` | Çekirdek interpreter (VM, types, builtins, modules) |
| `crates/monty-python/` | PyO3 Python bindings |
| `crates/monty-js/` | napi-rs JavaScript bindings |
| `crates/monty-cli/` | CLI aracı |
| `crates/monty-type-checking/` | Statik tip analizi |
| `crates/monty-typeshed/` | Tip stub dosyaları (vendor + custom) |
| `crates/fuzz/` | Fuzzing testleri |

### Modül whitelist

`import` ifadesi sadece şu modülleri yükleyebilir (kaynak: `modules/mod.rs`):

```
sys      typing      asyncio      pathlib      os
```

Başka herhangi bir `import X` → `ModuleNotFoundError`.

## 7. PydanticAI Entegrasyonu

Monty, PydanticAI'de **CodeMode** özelliğini güçlendirecek şekilde tasarlanmıştır. LLM sıralı tool çağrıları yapmak yerine, tool'ları fonksiyon olarak çağıran Python kodu yazar ve Monty bunu güvenli şekilde çalıştırır.

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

# Araçları tanımla
tools = FunctionToolset()

@tools.tool
async def get_weather(location: str) -> dict:
    ...

# Agent'ı CodeMode ile oluştur
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    toolsets=[CodeModeToolset(tools)],  # Monty-powered code execution
)

# Agent Python kodu yazarak tool'ları çağırır
result = await agent.run("Compare weather in London and Paris")
```

## 8. Alternatiflere Karşı Pozisyon

| Tech | Dil Tamamlığı | Güvenlik | Başlatma | Maliyet |
|------|---------------|----------|----------|---------|
| **Monty** | Kısmi | Katı | 0.06ms | Ücretsiz/OSS |
| Docker | Tam | İyi | 195ms | Ücretsiz/OSS |
| Pyodide | Tam | Zayıf | 2800ms | Ücretsiz/OSS |
| starlark-rust | Çok kısıtlı | İyi | 1.7ms | Ücretsiz/OSS |
| WASI/Wasmer | Neredeyse tam | Katı | 66ms | Ücretsiz* |
| Sandboxing servisi (E2B, Modal) | Tam | Katı | 1033ms | Ücretli |
| YOLO Python (exec) | Tam | Yok | 0.1ms | Ücretsiz/OSS |

**Monty'nin avantajları:** En düşük başlatma süresi + katı güvenlik + kolay kurulum + serileştirme desteği.

## 9. Eval Generation İçin Kullanım Senaryosu

### Problem

Eval generation pipeline'larında (hem tek seferlik generation hem de optimization döngüsünde) LLM agent expected değerleri **tahmin ediyor** — bu özellikle algoritmik fonksiyonlarda hallüsinasyona yol açar (ör. `binary_search([1,3,5,7], 5)` için yanlış index döndürme).

### Çözüm: CodeMode Eval Generation

CodeMode, **tüm eval generation pipeline'larında** kullanılabilecek genel bir mekanizmadır. Agent expected değerleri tahmin etmek yerine, Monty sandbox'ında **gerçek fonksiyonu çalıştırarak** ground-truth değerleri elde eder.

**Kullanım alanları:**
- **Tek seferlik eval generation** — `vowel` CLI veya API ile bir fonksiyon için eval dosyası üretirken
- **Optimization döngüsü** — GEPA ile prompt optimize ederken her iterasyonda (burada özellikle etkili çünkü yüzlerce eval üretiliyor)
- **CI/CD pipeline'ları** — Otomatik test üretimi akışlarında
- **Herhangi bir eval generation çağrısı** — CodeMode, pipeline'dan bağımsız bir altyapı katmanıdır

### Temel Mimari

```
┌─────────────────────────────────────────────────────────┐
│  LLM Agent                                              │
│  "Bu fonksiyon için ilginç test girdileri tasarla"      │
│                                                         │
│  Agent üretir:                                          │
│    inputs = [                                           │
│        {"x": [1,3,5,7,9], "target": 5},                │
│        {"x": [], "target": 1},                          │
│        {"x": [1], "target": 1},                         │
│    ]                                                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Monty Sandbox                                          │
│                                                         │
│  # Agent tarafından üretilen test harness               │
│  results = []                                           │
│  results.append(target_func([1,3,5,7,9], 5))           │
│  results.append(target_func([], 1))                     │
│  results.append(target_func([1], 1))                    │
│  results                                                │
│                                                         │
│  external_functions = {"target_func": real_function}    │
│  limits = ResourceLimits(max_duration_secs=5.0)         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Ground-Truth Sonuçlar                                  │
│                                                         │
│  results = [2, -1, 0]  ← gerçek fonksiyon çıktıları    │
│                                                         │
│  Bu değerler YAML eval dosyasındaki expected alanına    │
│  yazılır — hallüsinasyon riski sıfır.                   │
└─────────────────────────────────────────────────────────┘
```

### Neden External Function Mekanizması Kritik?

Hedef fonksiyon (ör. `binary_search`) şunları kullanabilir:
- Stdlib modülleri (`collections`, `itertools`, `math` vb.)
- Third-party kütüphaneler (`numpy`, `pandas` vb.)
- Dosya sistemi, network vb.

Monty sandbox'ı bunların hiçbirini desteklemez. **AMA** external function olarak inject edildiğinde, `target_func(...)` çağrısı host tarafındaki gerçek Python fonksiyonunu çalıştırır — yani tüm bağımlılıklar sorunsuz çalışır.

### ExecutorAdapter Protokolü (Taslak)

```python
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass
class ExecutionResult:
    """Sandbox çalıştırma sonucu."""
    output: Any                    # Kodun döndürdüğü değer
    stdout: str                    # print() çıktısı
    success: bool                  # Hatasız tamamlandı mı
    error: str | None = None       # Hata mesajı (varsa)
    error_type: str | None = None  # Hata tipi (ör. "ValueError")
    duration_ms: float = 0.0       # Çalışma süresi

class ExecutorAdapter(Protocol):
    """Kod çalıştırma adaptör protokolü."""
    async def execute(
        self,
        code: str,
        *,
        target_function: callable | None = None,
        inputs: dict[str, Any] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,  # 10MB
    ) -> ExecutionResult: ...

class MontyExecutor:
    """Monty tabanlı güvenli kod çalıştırıcı."""
    
    def __init__(self):
        import pydantic_monty
        self._monty = pydantic_monty
    
    async def execute(
        self,
        code: str,
        *,
        target_function: callable | None = None,
        inputs: dict[str, Any] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        import time
        
        stdout_lines: list[str] = []
        
        def print_callback(stream: str, text: str):
            stdout_lines.append(text)
        
        # External function listesi oluştur
        ext_names = ["target_func"] if target_function else []
        ext_impls = {"target_func": target_function} if target_function else {}
        
        # Input isimleri
        input_names = list(inputs.keys()) if inputs else []
        
        try:
            m = self._monty.Monty(
                code,
                inputs=input_names or None,
                external_functions=ext_names or None,
            )
            
            limits = self._monty.ResourceLimits(
                max_duration_secs=timeout,
                max_memory=max_memory,
            )
            
            start = time.perf_counter()
            result = m.run(
                inputs=inputs,
                limits=limits,
                external_functions=ext_impls,
                print_callback=print_callback,
            )
            duration = (time.perf_counter() - start) * 1000
            
            return ExecutionResult(
                output=result,
                stdout="\n".join(stdout_lines),
                success=True,
                duration_ms=duration,
            )
            
        except self._monty.MontyRuntimeError as e:
            inner = e.exception()
            return ExecutionResult(
                output=None,
                stdout="\n".join(stdout_lines),
                success=False,
                error=str(e),
                error_type=type(inner).__name__,
                duration_ms=0.0,
            )
        except self._monty.MontySyntaxError as e:
            return ExecutionResult(
                output=None,
                stdout="",
                success=False,
                error=str(e),
                error_type="SyntaxError",
                duration_ms=0.0,
            )
```

### BuiltinExecutor (Geliştirme/Fallback)

```python
class BuiltinExecutor:
    """exec() tabanlı çalıştırıcı — sadece güvenilir kodlar için."""
    
    async def execute(
        self,
        code: str,
        *,
        target_function: callable | None = None,
        inputs: dict[str, Any] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        import io, contextlib, time
        
        namespace = dict(inputs or {})
        if target_function:
            namespace["target_func"] = target_function
        
        stdout = io.StringIO()
        start = time.perf_counter()
        
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, namespace)
            duration = (time.perf_counter() - start) * 1000
            
            # Son ifadenin değerini al (eğer varsa)
            result = namespace.get("__result__", namespace.get("results"))
            
            return ExecutionResult(
                output=result,
                stdout=stdout.getvalue(),
                success=True,
                duration_ms=duration,
            )
        except Exception as e:
            return ExecutionResult(
                output=None,
                stdout=stdout.getvalue(),
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=0.0,
            )
```

## 10. Entegrasyon Tasarım Kararları

### Açık Sorular

1. **Agent'ın kodu nasıl üretecek?**
   - Seçenek A: Agent sadece input listesi üretir, harness kodu otomatik oluşturulur
   - Seçenek B: Agent tam test harness kodunu yazar (daha esnek ama hata riski daha yüksek)
   - Seçenek C: Hibrit — Agent input + beklenen davranış tanımlar, edge case'ler için raises testi de yazabilir

2. **Exception test etme nasıl olacak?**
   - `raises` assertion'ları için agent'ın exception beklediğini belirtmesi gerekir
   - Monty'de try/except destekleniyor, agent try/except yazarak exception tipini yakalayabilir

3. **Mevcut pipeline ile entegrasyon noktası neresi?**
   - `task.py`'daki `generate_and_score()` akışında, agent YAML ürettikten sonra expected değerleri doğrulamak için Monty kullanılabilir
   - Veya: Agent doğrudan Monty ile çalışan bir "CodeMode" prompt ile yönlendirilir

4. **Performans etkisi?**
   - Monty başlatma: ~0.06ms
   - Her test case çalıştırma: fonksiyonun karmaşıklığına bağlı (host'ta çalışır)
   - 25 fonksiyon × 20 test case = 500 çalıştırma → toplam <1 saniye ek maliyet

5. **Hangi fonksiyonlar CodeMode'a uygun?**
   - Deterministik fonksiyonlar (aynı input → aynı output): ✅ İdeal
   - Yan etkili fonksiyonlar (dosya yazma, API çağrısı): ⚠️ Dikkatli olunmalı
   - Rastgele çıktılı fonksiyonlar: ❌ Uygun değil (expected value sabitlenmeli)

### Kısıtlamalar ve Çözümler

| Kısıtlama | Etki | Çözüm |
|-----------|------|-------|
| Class tanımı yok | Agent class kullanamaz | Fonksiyon + dict / NamedTuple kullan |
| `json` modülü yok | String serialization zor | Host'a external function olarak delege et |
| `match` statement yok | Pattern matching yok | if/elif zincirleri kullan |
| `with` statement yok | Context manager yok | İstisnai durum; hedef fonksiyon host'ta çalışır |
| `math`, `collections`, `itertools` yok | Sandbox içi hesaplama kısıtlı | Tüm asıl hesaplama host fonksiyonunda yapılır |
| Sadece 5 modül import edilebilir | `sys`, `typing`, `asyncio`, `pathlib`, `os` | Yeterli — sandbox kodu sadece orkestrasyon yapıyor |

**En kritik çözüm:** Sandbox kodunun amacı karmaşık hesaplama yapmak değil — **sadece test girdilerini organize edip hedef fonksiyonu çağırmak**. Asıl hesaplama external function (hedef fonksiyon) içinde, host tarafında yapılır.

## 11. Örnek: Tam Çalışma Akışı

```python
# 1. Hedef fonksiyon (test edilecek)
def binary_search(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# 2. Agent'ın ürettiği Monty kodu
agent_code = """
results = []

# Normal cases
results.append({"input": {"arr": [1,3,5,7,9], "target": 5}, "expected": target_func([1,3,5,7,9], 5)})
results.append({"input": {"arr": [1,3,5,7,9], "target": 1}, "expected": target_func([1,3,5,7,9], 1)})
results.append({"input": {"arr": [1,3,5,7,9], "target": 9}, "expected": target_func([1,3,5,7,9], 9)})

# Not found
results.append({"input": {"arr": [1,3,5,7,9], "target": 4}, "expected": target_func([1,3,5,7,9], 4)})

# Edge cases
results.append({"input": {"arr": [], "target": 1}, "expected": target_func([], 1)})
results.append({"input": {"arr": [1], "target": 1}, "expected": target_func([1], 1)})
results.append({"input": {"arr": [1], "target": 2}, "expected": target_func([1], 2)})

results
"""

# 3. Monty'de çalıştır
import pydantic_monty

m = pydantic_monty.Monty(
    agent_code,
    external_functions=["target_func"],
)

results = m.run(
    external_functions={"target_func": binary_search},
    limits=pydantic_monty.ResourceLimits(max_duration_secs=5.0),
)

# 4. Sonuç: Ground-truth expected değerlerle test case'ler
# results = [
#     {"input": {"arr": [1,3,5,7,9], "target": 5}, "expected": 2},
#     {"input": {"arr": [1,3,5,7,9], "target": 1}, "expected": 0},
#     {"input": {"arr": [1,3,5,7,9], "target": 9}, "expected": 4},
#     {"input": {"arr": [1,3,5,7,9], "target": 4}, "expected": -1},
#     {"input": {"arr": [], "target": 1}, "expected": -1},
#     {"input": {"arr": [1], "target": 1}, "expected": 0},
#     {"input": {"arr": [1], "target": 2}, "expected": -1},
# ]
```

**Hiçbir expected değer hallüsine edilmedi — hepsi gerçek fonksiyon çıktısı.**

## 12. Sonraki Adımlar

1. ~~Monty API'yi tam anla~~ ✅
2. `ExecutorAdapter` protokolünü finalize et
3. `MontyExecutor` implementasyonunu yaz
4. `task.py`'ye CodeMode akışını entegre et
5. Agent prompt'unu CodeMode için güncelle
6. 25 referans fonksiyon üzerinde test et
7. Mevcut "tahmin" modu ile CodeMode'u karşılaştır (A/B)
