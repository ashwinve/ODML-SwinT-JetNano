# %%
import tarfile, zipfile
tarf = tarfile.open( name='./imagenet/tar_archives/test_images.tar.gz', mode='r|gz' )
zipf = zipfile.ZipFile( file='./imagenet/zipped_archives/test.zip', mode='a', compression=zipfile.ZIP_DEFLATED )
for m in tarf:
    f = tarf.extractfile( m )
    fl = f.read()
    fn = m.name
    zipf.writestr( fn, fl )
tarf.close()
zipf.close()
# %%
import tarfile, zipfile
tarf = tarfile.open( name='./imagenet/tar_archives/val_images.tar.gz', mode='r|gz' )
zipf = zipfile.ZipFile( file='./imagenet/zipped_archives/val.zip', mode='a', compression=zipfile.ZIP_DEFLATED )
for m in tarf:
    f = tarf.extractfile( m )
    fl = f.read()
    fn = m.name
    zipf.writestr( fn, fl )
tarf.close()
zipf.close()
# %%
