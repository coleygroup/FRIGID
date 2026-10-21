import torch
from test_mist_checkpoint import release as release
from test_mist_checkpoint import spectrum_batch

from dlm.utils.checkpoint import pack_checkpoint
from mist.models.scaled import ScaledMIST


def test_unified_checkpoint_preserves_mist_architecture_and_decoder(release, tmp_path):
    path, reference, _ = release
    decoder_path = tmp_path / 'decoder.ckpt'
    output = tmp_path / 'combined.ckpt'
    decoder = {'hyper_parameters': {'hidden_size': 17}, 'state_dict': {'weight': torch.arange(17)}}
    torch.save(decoder, decoder_path)
    pack_checkpoint(str(path), str(decoder_path), str(output))
    combined = torch.load(output, weights_only=False)
    assert combined['encoder_type'] == 'scaled_mist'
    assert combined['encoder_config']['hidden_size'] == 32
    assert combined['decoder_checkpoint']['hyper_parameters'] == decoder['hyper_parameters']
    torch.testing.assert_close(combined['decoder_checkpoint']['state_dict']['weight'], decoder['state_dict']['weight'])
    restored = ScaledMIST(**combined['encoder_config']).eval()
    restored.load_state_dict(combined['encoder_state_dict'], strict=True)
    with torch.inference_mode():
        expected, _ = reference(spectrum_batch())
        actual, _ = restored(spectrum_batch())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
