"""_get_credentials resilience: a dead refresh token falls back to consent.

An OAuth app in "Testing" status has its refresh tokens expired by Google after
7 days; refreshing then raises RefreshError. _get_credentials must discard the
dead token and re-run the consent flow instead of propagating invalid_grant.
"""

from unittest.mock import MagicMock, patch

from google.auth.exceptions import RefreshError

import calendar_reader


def _stale_creds():
    creds = MagicMock()
    creds.valid = False
    creds.expired = True
    creds.refresh_token = "rt"
    creds.refresh.side_effect = RefreshError("invalid_grant: Token has been expired or revoked.")
    return creds


@patch("calendar_reader.open")
@patch("calendar_reader._run_consent_flow")
@patch("calendar_reader.Credentials.from_authorized_user_file")
@patch("calendar_reader.os.path.exists", return_value=True)
def test_revoked_refresh_token_falls_back_to_consent(mock_exists, mock_from_file, mock_consent, mock_open):
    mock_from_file.return_value = _stale_creds()
    fresh = MagicMock()
    fresh.to_json.return_value = "{}"
    mock_consent.return_value = fresh

    creds = calendar_reader._get_credentials()

    mock_consent.assert_called_once()
    assert creds is fresh


@patch("calendar_reader.open")
@patch("calendar_reader._run_consent_flow")
@patch("calendar_reader.Request")
@patch("calendar_reader.Credentials.from_authorized_user_file")
@patch("calendar_reader.os.path.exists", return_value=True)
def test_valid_refresh_does_not_reconsent(mock_exists, mock_from_file, mock_request, mock_consent, mock_open):
    creds = MagicMock()
    creds.valid = False
    creds.expired = True
    creds.refresh_token = "rt"
    creds.refresh.side_effect = None  # refresh succeeds
    creds.to_json.return_value = "{}"
    mock_from_file.return_value = creds

    result = calendar_reader._get_credentials()

    creds.refresh.assert_called_once()
    mock_consent.assert_not_called()
    assert result is creds


@patch("calendar_reader.open")
@patch("calendar_reader._run_consent_flow")
@patch("calendar_reader.os.path.exists", return_value=False)
def test_no_token_file_runs_consent(mock_exists, mock_consent, mock_open):
    fresh = MagicMock()
    fresh.to_json.return_value = "{}"
    mock_consent.return_value = fresh

    result = calendar_reader._get_credentials()

    mock_consent.assert_called_once()
    assert result is fresh
