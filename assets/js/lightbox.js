(function () {
  var imgs = document.querySelectorAll('.prose img');
  if (!imgs.length) return;
  var dlg = document.createElement('dialog');
  dlg.className = 'lightbox';
  dlg.setAttribute('aria-label', 'Enlarged image');
  var big = document.createElement('img');
  var btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'lightbox-close';
  btn.setAttribute('aria-label', 'Close'); btn.textContent = '×';
  dlg.append(big, btn);
  document.body.appendChild(dlg);
  var zoomed = false;
  function reset() { zoomed = false; dlg.classList.remove('zoomed'); }
  function open(img) {
    big.src = img.currentSrc || img.src;
    big.alt = img.alt;
    reset();
    dlg.showModal();
  }
  imgs.forEach(function (img) {
    img.classList.add('zoomable');
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.addEventListener('click', function () { open(img); });
    img.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(img); }
    });
  });
  // Click image: toggle full-size (scrollable). Click backdrop or x: close.
  big.addEventListener('click', function (e) {
    e.stopPropagation();
    zoomed = !zoomed;
    dlg.classList.toggle('zoomed', zoomed);
  });
  dlg.addEventListener('click', function () { dlg.close(); });
  btn.addEventListener('click', function () { dlg.close(); });
  dlg.addEventListener('close', reset);
})();
