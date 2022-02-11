using System;
using CTCBeamDecoder.CTCBeamDecoderInterop;

namespace CTCBeamDecoder.Models;

/// <summary>
/// Class using for maintain different chunks of data in one beam algorithm corresponding to one unique source.
/// </summary>
public sealed class DecoderState : IDisposable
{
    internal IntPtr State;
    private bool _disposedValue;

    internal DecoderState(IntPtr state)
    {
        State = state;
    }

    private void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            CTCDecoderInterop.ReleaseState(State);
            _disposedValue = true;
        }
    }

    ~DecoderState()
    {
        Dispose(disposing: false);
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
