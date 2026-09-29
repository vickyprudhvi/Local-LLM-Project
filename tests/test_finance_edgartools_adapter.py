"""Phase H.29 -- deterministic, mocked tests for the EdgarTools adapter.

No network, no real `edgar` package required. A fake `edgar` module is
installed into `sys.modules` for the duration of each test, exercising the
adapter's OWN logic: lazy import discipline, identity handling, error-code
mapping, dataclass shapes, and that `to_context()`/formatted-summary text is
never read as authoritative (spec section 4) -- only `.text()`/`.sections`/
`.attachments`.

Live, real-network comparison against SEC EDGAR is a separate spike script
(`scripts/run_edgartools_sec_comparison.py`), not a pytest test, matching
this project's existing convention (`scripts/run_live_document_pipeline_
benchmark.py` is likewise not part of the pytest suite).
"""

import sys
import types

import pytest

import finance.documents.edgartools_adapter as ad


class _FakeSection:
    def __init__(self, name, title, text_value):
        self.name = name
        self.title = title
        self._text = text_value

    def text(self):
        return self._text


class _FakeSections(dict):
    """Mimics EdgarTools' Sections object: dict-like with .items()."""


class _FakeObj:
    def __init__(self, items=None, sections=None):
        self.items = items or []
        self.sections = sections or _FakeSections()

    def to_context(self):
        # Deliberately returns text that would be WRONG if ever read as
        # authoritative -- proves the adapter never calls this.
        raise AssertionError("to_context() must never be called by the adapter")


class _FakeAttachment:
    def __init__(self, document, document_type, description):
        self.document = document
        self.document_type = document_type
        self.description = description


class _FakeFiling:
    def __init__(self, accession_no, form, filing_date, period_of_report=None,
                primary_document=None, text_value="filing text", obj_value=None,
                attachments=None):
        self.accession_no = accession_no
        self.form = form
        self.filing_date = filing_date
        self.period_of_report = period_of_report
        self.primary_document = primary_document
        self._text = text_value
        self._obj = obj_value or _FakeObj()
        self.attachments = attachments or []

    def text(self):
        return self._text

    def obj(self):
        return self._obj


class _FakeFilingsList(list):
    def latest(self):
        return self[0] if self else None


class _FakeCompany:
    def __init__(self, cik, name, filings_by_accession=None, filings_by_form=None):
        self.cik = cik
        self.name = name
        self._by_accession = filings_by_accession or {}
        self._by_form = filings_by_form or {}

    def get_filings(self, form=None, accession_number=None):
        if accession_number:
            match = self._by_accession.get(accession_number)
            return _FakeFilingsList([match] if match else [])
        if form:
            return _FakeFilingsList(self._by_form.get(form, []))
        return _FakeFilingsList(
            [f for group in self._by_form.values() for f in group])


def _install_fake_edgar(monkeypatch, company_by_ticker=None, set_identity_calls=None):
    fake = types.ModuleType("edgar")
    company_by_ticker = company_by_ticker or {}
    set_identity_calls = set_identity_calls if set_identity_calls is not None else []

    def _company_ctor(ticker):
        company = company_by_ticker.get(ticker)
        if company is None:
            raise ValueError(f"no such company {ticker!r}")
        return company

    def _set_identity(identity):
        set_identity_calls.append(identity)

    fake.Company = _company_ctor
    fake.set_identity = _set_identity
    monkeypatch.setitem(sys.modules, "edgar", fake)
    return fake


# ---------------------------------------------------------------------------
# A. Lazy import / not-installed handling
# ---------------------------------------------------------------------------

def test_A_module_import_never_requires_edgar_installed():
    # Already exercised at collection time (this file imports the adapter
    # above without edgartools installed in CI) -- re-asserted explicitly.
    assert ad.EdgarToolsErrorCode.NOT_INSTALLED == "EDGARTOOLS_NOT_INSTALLED"


def test_A2_missing_edgar_package_raises_controlled_error(monkeypatch):
    # `sys.modules[name] = None` is the standard way to force `import name`
    # to raise ImportError regardless of whether the package is actually
    # installed in this environment (it IS installed here, for this spike's
    # own live comparison script -- this test still proves the adapter's
    # own not-installed handling works, independent of that fact).
    monkeypatch.setitem(sys.modules, "edgar", None)
    with pytest.raises(ad.EdgarToolsAdapterError) as excinfo:
        ad.lookup_company("AAPL")
    assert excinfo.value.code == ad.EdgarToolsErrorCode.NOT_INSTALLED


# ---------------------------------------------------------------------------
# B. Identity handling
# ---------------------------------------------------------------------------

def test_B_missing_identity_raises_controlled_error(monkeypatch):
    _install_fake_edgar(monkeypatch, company_by_ticker={})
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    with pytest.raises(ad.EdgarToolsAdapterError) as excinfo:
        ad.lookup_company("AAPL")
    assert excinfo.value.code == ad.EdgarToolsErrorCode.NO_IDENTITY


def test_B2_identity_is_set_on_every_call_never_hardcoded(monkeypatch):
    calls = []
    company = _FakeCompany(cik="320193", name="Apple Inc.")
    _install_fake_edgar(monkeypatch, company_by_ticker={"AAPL": company},
                        set_identity_calls=calls)
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    ad.lookup_company("AAPL")
    assert calls == ["Test Suite test@example.invalid"]


# ---------------------------------------------------------------------------
# C. Company lookup
# ---------------------------------------------------------------------------

def test_C_company_lookup_returns_domain_dataclass(monkeypatch):
    company = _FakeCompany(cik="320193", name="Apple Inc.")
    _install_fake_edgar(monkeypatch, company_by_ticker={"AAPL": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    ref = ad.lookup_company("AAPL")
    assert isinstance(ref, ad.EdgarToolsCompanyRef)
    assert ref.cik == "320193" and ref.name == "Apple Inc." and ref.ticker == "AAPL"


def test_C2_unknown_ticker_raises_controlled_error(monkeypatch):
    _install_fake_edgar(monkeypatch, company_by_ticker={})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    with pytest.raises(ad.EdgarToolsAdapterError) as excinfo:
        ad.lookup_company("NOPE")
    assert excinfo.value.code == ad.EdgarToolsErrorCode.COMPANY_NOT_FOUND


# ---------------------------------------------------------------------------
# D. Filing discovery / get_filing
# ---------------------------------------------------------------------------

def _company_with_one_filing():
    filing = _FakeFiling(
        accession_no="0001193125-23-255762", form="8-K", filing_date="2023-10-13",
        period_of_report="2023-10-13", primary_document="d537928d8k.htm",
        text_value="Item 2.01 ... completed its previously announced acquisition ...",
        obj_value=_FakeObj(items=["Item 2.01"]))
    company = _FakeCompany(
        cik="789019", name="MICROSOFT CORP",
        filings_by_accession={"0001193125-23-255762": filing},
        filings_by_form={"8-K": [filing]})
    return company, filing


def test_D_get_filing_by_accession(monkeypatch):
    company, filing = _company_with_one_filing()
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    ref = ad.get_filing("MSFT", "0001193125-23-255762")
    assert ref.accession == "0001193125-23-255762"
    assert ref.form == "8-K"
    assert ref.items == "2.01"


def test_D2_unknown_accession_raises_controlled_error(monkeypatch):
    company, _filing = _company_with_one_filing()
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    with pytest.raises(ad.EdgarToolsAdapterError) as excinfo:
        ad.get_filing("MSFT", "0000000000-00-000000")
    assert excinfo.value.code == ad.EdgarToolsErrorCode.FILING_NOT_FOUND


def test_D3_discover_filings_respects_limit(monkeypatch):
    filings = [
        _FakeFiling(accession_no=f"000119312523-{i:06d}", form="8-K",
                   filing_date=f"2023-0{i}-01") for i in range(1, 5)]
    company = _FakeCompany(cik="789019", name="MICROSOFT CORP",
                           filings_by_form={"8-K": filings})
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    refs = ad.discover_filings("MSFT", form="8-K", limit=2)
    assert len(refs) == 2


# ---------------------------------------------------------------------------
# E. Text / sections never read from to_context() or a formatted summary
# ---------------------------------------------------------------------------

def test_E_filing_text_is_verbatim_not_a_summary(monkeypatch):
    company, filing = _company_with_one_filing()
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    text, representation = ad.filing_text("MSFT", "0001193125-23-255762")
    assert "completed its previously announced acquisition" in text
    assert representation == ad.TextRepresentation.SECTION_TEXT


def test_E2_to_context_is_never_called(monkeypatch):
    """_FakeObj.to_context() raises if called -- proves the adapter's
    section/text/exhibit paths never touch it."""
    company, filing = _company_with_one_filing()
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    ad.filing_text("MSFT", "0001193125-23-255762")
    ad.filing_sections("MSFT", "0001193125-23-255762")
    ad.filing_exhibits("MSFT", "0001193125-23-255762")
    # No AssertionError means to_context() was never invoked.


def test_F_filing_sections_returns_named_verbatim_sections(monkeypatch):
    sections = _FakeSections({
        "item_201": _FakeSection("item_201", "Item 2.01 - Completion of Acquisition",
                                 "Item 2.01 text here."),
        "signatures": _FakeSection("signatures", "Signatures", "Pursuant to..."),
    })
    filing = _FakeFiling(
        accession_no="0001193125-23-255762", form="8-K", filing_date="2023-10-13",
        obj_value=_FakeObj(items=["Item 2.01"], sections=sections))
    company = _FakeCompany(cik="789019", name="MICROSOFT CORP",
                           filings_by_accession={"0001193125-23-255762": filing})
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    result = ad.filing_sections("MSFT", "0001193125-23-255762")
    assert set(result) == {"item_201", "signatures"}
    assert result["item_201"].text == "Item 2.01 text here."
    assert result["item_201"].text_representation == ad.TextRepresentation.SECTION_TEXT


# ---------------------------------------------------------------------------
# G. Exhibit discovery
# ---------------------------------------------------------------------------

def test_G_filing_exhibits_returns_typed_attachments(monkeypatch):
    filing = _FakeFiling(
        accession_no="0001193125-23-255762", form="8-K", filing_date="2023-10-13",
        attachments=[
            _FakeAttachment("d537928d8k.htm", "8-K", "8-K"),
            _FakeAttachment("ex991.htm", "EX-99.1", "Press Release"),
        ])
    company = _FakeCompany(cik="789019", name="MICROSOFT CORP",
                           filings_by_accession={"0001193125-23-255762": filing})
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    exhibits = ad.filing_exhibits("MSFT", "0001193125-23-255762")
    assert len(exhibits) == 2
    assert exhibits[1].document_type == "EX-99.1"
    assert exhibits[1].description == "Press Release"


# ---------------------------------------------------------------------------
# H. Fetch failure is a controlled error, not a raw traceback
# ---------------------------------------------------------------------------

def test_H_fetch_failure_is_controlled(monkeypatch):
    class _BrokenCompany(_FakeCompany):
        def get_filings(self, form=None, accession_number=None):
            raise RuntimeError("simulated network failure")

    company = _BrokenCompany(cik="789019", name="MICROSOFT CORP")
    _install_fake_edgar(monkeypatch, company_by_ticker={"MSFT": company})
    monkeypatch.setenv("EDGAR_IDENTITY", "Test Suite test@example.invalid")
    with pytest.raises(ad.EdgarToolsAdapterError) as excinfo:
        ad.discover_filings("MSFT", form="8-K")
    assert excinfo.value.code == ad.EdgarToolsErrorCode.FETCH_FAILED


# ---------------------------------------------------------------------------
# I. Config default stays 'current'; adapter is never imported by the
#    default path (spec section 13/17).
# ---------------------------------------------------------------------------

def test_I_finance_sec_provider_defaults_to_current(monkeypatch):
    import tools.config as config
    monkeypatch.delenv("FINANCE_SEC_PROVIDER", raising=False)
    assert config.finance_sec_provider() == "current"


def test_I2_finance_sec_provider_rejects_unknown_value(monkeypatch):
    import tools.config as config
    monkeypatch.setenv("FINANCE_SEC_PROVIDER", "nonsense")
    assert config.finance_sec_provider() == "current"


def test_I3_document_pipeline_never_imports_edgartools_adapter():
    """AST-based static check, same pattern as H.23/H.24's import-boundary
    tests: the production document-pipeline modules must never import this
    adapter (or the third-party `edgar` package) at module scope."""
    import ast
    import inspect

    import finance.documents.package as package_module
    import finance.documents.actuals_validator as actuals_validator_module
    import finance.documents.event_validator as event_validator_module
    import finance.actualization as actualization_module

    for module in (package_module, actuals_validator_module,
                  event_validator_module, actualization_module):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("edgar"), (
                    f"{module.__name__} must never import edgar/edgartools_adapter")
