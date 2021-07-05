export var azAZ = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Əvvəlki səhifə',
      labelRowsPerPage: 'Səhifəyə düşən sətrlər:',
      labelDisplayedRows: function labelDisplayedRows(_ref) {
        var from = _ref.from,
            to = _ref.to,
            count = _ref.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " d\u0259n ").concat(count);
      },
      nextIconButtonText: 'Növbəti səhifə'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        var pluralForm = 'Ulduz';
        var lastDigit = value % 10;

        if (lastDigit > 1 && lastDigit < 5) {
          pluralForm = 'Ulduzlar';
        }

        return "".concat(value, " ").concat(pluralForm);
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Silmək',
      closeText: 'Bağlamaq',
      loadingText: 'Yüklənir…',
      noOptionsText: 'Seçimlər mövcud deyil',
      openText: 'Открыть'
    },
    MuiAlert: {
      closeText: 'Bağlamaq'
    }
  }
};
export var bgBG = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Предишна страница',
      labelRowsPerPage: 'Редове на страница:',
      labelDisplayedRows: function labelDisplayedRows(_ref2) {
        var from = _ref2.from,
            to = _ref2.to,
            count = _ref2.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " \u043E\u0442 ").concat(count);
      },
      nextIconButtonText: 'Следваща страница'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " \u0417\u0432\u0435\u0437\u0434").concat(value !== 1 ? 'и' : 'а');
      },
      emptyLabelText: 'Изчисти'
    },
    MuiAutocomplete: {
      clearText: 'Изчисти',
      closeText: 'Затвори',
      loadingText: 'Зареждане…',
      noOptionsText: 'Няма налични опции',
      openText: 'Отвори'
    },
    MuiAlert: {
      closeText: 'Затвори'
    }
  }
};
export var caES = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Pàgina anterior',
      labelRowsPerPage: 'Files per pàgina:',
      labelDisplayedRows: function labelDisplayedRows(_ref3) {
        var from = _ref3.from,
            to = _ref3.to,
            count = _ref3.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " de ").concat(count);
      },
      nextIconButtonText: 'Següent pàgina'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " ").concat(value !== 1 ? 'Estrelles' : 'Estrella');
      },
      emptyLabelText: 'Buit'
    },
    MuiAutocomplete: {
      clearText: 'Netejar',
      closeText: 'Tancar',
      loadingText: 'Carregant…',
      noOptionsText: 'Sense opcions',
      openText: 'Obert'
    },
    MuiAlert: {
      closeText: 'Tancat'
    }
  }
};
export var csCZ = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Předchozí stránka',
      labelRowsPerPage: 'Řádků na stránce:',
      labelDisplayedRows: function labelDisplayedRows(_ref4) {
        var from = _ref4.from,
            to = _ref4.to,
            count = _ref4.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " z ").concat(count);
      },
      nextIconButtonText: 'Další stránka'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        if (value === 1) {
          return "".concat(value, " hv\u011Bzdi\u010Dka");
        }

        if (value >= 2 && value <= 4) {
          return "".concat(value, " hv\u011Bzdi\u010Dky");
        }

        return "".concat(value, " hv\u011Bzdi\u010Dek");
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Vymazat',
      closeText: 'Zavřít',
      loadingText: 'Načítání…',
      noOptionsText: 'Žádné možnosti',
      openText: 'Otevřít'
    },
    MuiAlert: {
      closeText: 'Zavřít'
    }
  }
};
export var deDE = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Nächste Seite',
      labelRowsPerPage: 'Zeilen pro Seite:',
      labelDisplayedRows: function labelDisplayedRows(_ref5) {
        var from = _ref5.from,
            to = _ref5.to,
            count = _ref5.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " von ").concat(count);
      },
      nextIconButtonText: 'Nächste Seite'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " ").concat(value !== 1 ? 'Sterne' : 'Stern');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Leeren',
      closeText: 'Schließen',
      loadingText: 'Wird geladen…',
      noOptionsText: 'Keine Optionen',
      openText: 'Öffnen'
    },
    MuiAlert: {
      closeText: 'Schließen'
    }
  }
}; // default

export var enUS = {
  /**
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Previous page',
      labelRowsPerPage: 'Rows per page:',
      labelDisplayedRows: ({ from, to, count }) => `${from}-${to === -1 ? count : to} of ${count}`,
      nextIconButtonText: 'Next page',
    },
    MuiRating: {
      getLabelText: value => `${value} Star${value !== 1 ? 's' : ''}`,
      emptyLabelText: 'Empty',
    },
    MuiAutocomplete: {
      clearText: 'Clear',
      closeText: 'Close',
      loadingText: 'Loading…',
      noOptionsText: 'No options',
      openText: 'Open',
    },
    MuiAlert: {
      closeText: 'Close',
    },
  },
  */
};
export var esES = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Página anterior',
      labelRowsPerPage: 'Filas por página:',
      labelDisplayedRows: function labelDisplayedRows(_ref6) {
        var from = _ref6.from,
            to = _ref6.to,
            count = _ref6.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " de ").concat(count);
      },
      nextIconButtonText: 'Siguiente página'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Estrella").concat(value !== 1 ? 's' : '');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Limpiar',
      closeText: 'Cerrar',
      loadingText: 'Cargando…',
      noOptionsText: 'Sin opciones',
      openText: 'Abierto'
    },
    MuiAlert: {
      closeText: 'Cerrar'
    }
  }
};
export var etEE = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Eelmine lehekülg',
      labelRowsPerPage: 'Ridu leheküljel:',
      labelDisplayedRows: function labelDisplayedRows(_ref7) {
        var from = _ref7.from,
            to = _ref7.to,
            count = _ref7.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " / ").concat(count);
      },
      nextIconButtonText: 'Järgmine lehekülg'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " T\xE4rn").concat(value !== 1 ? 'i' : '');
      },
      emptyLabelText: 'Tühi'
    },
    MuiAutocomplete: {
      clearText: 'Tühjenda',
      closeText: 'Sulge',
      loadingText: 'Laen…',
      noOptionsText: 'Valikuid ei ole',
      openText: 'Ava'
    },
    MuiAlert: {
      closeText: 'Sulge'
    },
    MuiPagination: {
      'aria-label': 'Lehekülgede valik',
      getItemAriaLabel: function getItemAriaLabel(type, page, selected) {
        if (type === 'page') {
          return "".concat(selected ? '' : 'Vali ', "lehek\xFClg ").concat(page);
        }

        if (type === 'first') {
          return 'Vali esimene lehekülg';
        }

        if (type === 'last') {
          return 'Vali viimane lehekülg';
        }

        if (type === 'next') {
          return 'Vali järgmine lehekülg';
        }

        if (type === 'previous') {
          return 'Vali eelmine lehekülg';
        }

        return undefined;
      }
    }
  }
};
export var faIR = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'صفحهٔ قبل',
      labelRowsPerPage: 'تعداد سطرهای هر صفحه:',
      labelDisplayedRows: function labelDisplayedRows(_ref8) {
        var from = _ref8.from,
            to = _ref8.to,
            count = _ref8.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " \u0627\u0632 ").concat(count);
      },
      nextIconButtonText: 'صفحهٔ بعد'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " \u0633\u062A\u0627\u0631\u0647");
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'پاک‌کردن',
      closeText: 'بستن',
      loadingText: 'در حال بارگذاری…',
      noOptionsText: 'بی‌نتیجه',
      openText: 'بازکردن'
    },
    MuiAlert: {
      closeText: 'بستن'
    }
  }
};
export var fiFI = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Edellinen sivu',
      labelRowsPerPage: 'Rivejä per sivu:',
      labelDisplayedRows: function labelDisplayedRows(_ref9) {
        var from = _ref9.from,
            to = _ref9.to,
            count = _ref9.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " / ").concat(count);
      },
      nextIconButtonText: 'Seuraava sivu'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " T\xE4ht").concat(value !== 1 ? 'eä' : 'i');
      },
      emptyLabelText: 'Tyhjä'
    },
    MuiAutocomplete: {
      clearText: 'Tyhjennä',
      closeText: 'Sulje',
      loadingText: 'Ladataan…',
      noOptionsText: 'Ei valintoja',
      openText: 'Avaa'
    },
    MuiAlert: {
      closeText: 'Sulje'
    }
  }
};
export var frFR = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Page précédente',
      labelRowsPerPage: 'Lignes par page :',
      labelDisplayedRows: function labelDisplayedRows(_ref10) {
        var from = _ref10.from,
            to = _ref10.to,
            count = _ref10.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " sur ").concat(count);
      },
      nextIconButtonText: 'Page suivante'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Etoile").concat(value !== 1 ? 's' : '');
      },
      emptyLabelText: 'Vide'
    },
    MuiAutocomplete: {
      clearText: 'Vider',
      closeText: 'Fermer',
      loadingText: 'Chargement…',
      noOptionsText: 'Pas de résultats',
      openText: 'Ouvrir'
    },
    MuiAlert: {
      closeText: 'Fermer'
    },
    MuiPagination: {
      'aria-label': 'pagination navigation',
      getItemAriaLabel: function getItemAriaLabel(type, page, selected) {
        if (type === 'page') {
          return "".concat(selected ? '' : 'Aller à la ', "page ").concat(page);
        }

        if (type === 'first') {
          return 'Aller à la première page';
        }

        if (type === 'last') {
          return 'Aller à la dernière page';
        }

        if (type === 'next') {
          return 'Aller à la page suivante';
        }

        if (type === 'previous') {
          return 'Aller à la page précédente';
        }

        return undefined;
      }
    }
  }
};
export var huHU = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Előző oldal',
      labelRowsPerPage: 'Sorok száma:',
      labelDisplayedRows: function labelDisplayedRows(_ref11) {
        var from = _ref11.from,
            to = _ref11.to,
            count = _ref11.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " / ").concat(count);
      },
      nextIconButtonText: 'Következő oldal'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Csillag");
      },
      emptyLabelText: 'Üres'
    },
    MuiAutocomplete: {
      clearText: 'Törlés',
      closeText: 'Bezárás',
      loadingText: 'Töltés…',
      noOptionsText: 'Nincs találat',
      openText: 'Megnyitás'
    },
    MuiAlert: {
      closeText: 'Bezárás'
    }
  }
};
export var idID = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Halaman sebelumnya',
      labelRowsPerPage: 'Baris per halaman:',
      labelDisplayedRows: function labelDisplayedRows(_ref12) {
        var from = _ref12.from,
            to = _ref12.to,
            count = _ref12.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " dari ").concat(count);
      },
      nextIconButtonText: 'Halaman selanjutnya'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Bintang");
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Hapus',
      closeText: 'Tutup',
      loadingText: 'Memuat…',
      noOptionsText: 'Tidak ada opsi',
      openText: 'Buka'
    },
    MuiAlert: {
      closeText: 'Tutup'
    }
  }
};
export var isIS = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Fyrri síða',
      labelRowsPerPage: 'Raðir á síðu:',
      labelDisplayedRows: function labelDisplayedRows(_ref13) {
        var from = _ref13.from,
            to = _ref13.to,
            count = _ref13.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " af ").concat(count);
      },
      nextIconButtonText: 'Næsta síða'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " ").concat(value === 1 ? 'Stjarna' : 'Stjörnur');
      },
      emptyLabelText: 'Tómt'
    },
    MuiAutocomplete: {
      clearText: 'Hreinsa',
      closeText: 'Loka',
      loadingText: 'Hlaða…',
      noOptionsText: 'Engar niðurstöður',
      openText: 'Opna'
    },
    MuiAlert: {
      closeText: 'Loka'
    }
  }
};
export var itIT = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Pagina precedente',
      labelRowsPerPage: 'Righe per pagina:',
      labelDisplayedRows: function labelDisplayedRows(_ref14) {
        var from = _ref14.from,
            to = _ref14.to,
            count = _ref14.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " di ").concat(count);
      },
      nextIconButtonText: 'Pagina successiva'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Stell").concat(value !== 1 ? 'e' : 'a');
      },
      emptyLabelText: 'Vuoto'
    },
    MuiAutocomplete: {
      clearText: 'Svuota',
      closeText: 'Chiudi',
      loadingText: 'Caricamento in corso…',
      noOptionsText: 'Nessuna opzione',
      openText: 'Apri'
    },
    MuiAlert: {
      closeText: 'Chiudi'
    }
  }
};
export var jaJP = {
  props: {
    MuiTablePagination: {
      backIconButtonText: '前のページ',
      labelRowsPerPage: 'ページごとの行:',
      labelDisplayedRows: function labelDisplayedRows(_ref15) {
        var from = _ref15.from,
            to = _ref15.to,
            count = _ref15.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " of ").concat(count);
      },
      nextIconButtonText: '次のページ'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " ").concat(value !== 1 ? '出演者' : '星');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'クリア',
      closeText: '閉じる',
      loadingText: '積み込み…',
      noOptionsText: '結果がありません',
      openText: '開いた'
    },
    MuiAlert: {
      closeText: '閉じる'
    }
  }
};
export var koKR = {
  props: {
    MuiTablePagination: {
      backIconButtonText: '이전 페이지',
      labelRowsPerPage: '페이지 당 행:',
      labelDisplayedRows: function labelDisplayedRows(_ref16) {
        var from = _ref16.from,
            to = _ref16.to,
            count = _ref16.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " / ").concat(count);
      },
      nextIconButtonText: '다음 페이지'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " \uC810");
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: '지우기',
      closeText: '닫기',
      loadingText: '불러오는 중…',
      noOptionsText: '옵션 없음',
      openText: '열기'
    }
  }
};
export var nlNL = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Vorige pagina',
      labelRowsPerPage: 'Regels per pagina :',
      labelDisplayedRows: function labelDisplayedRows(_ref17) {
        var from = _ref17.from,
            to = _ref17.to,
            count = _ref17.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " van ").concat(count);
      },
      nextIconButtonText: 'Volgende pagina'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Ster").concat(value !== 1 ? 'ren' : '');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Wissen',
      closeText: 'Sluiten',
      loadingText: 'Laden…',
      noOptionsText: 'Geen opties',
      openText: 'Openen'
    },
    MuiAlert: {
      closeText: 'Sluiten'
    }
  }
};
export var plPL = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Poprzednia strona',
      labelRowsPerPage: 'Wierszy na stronę:',
      labelDisplayedRows: function labelDisplayedRows(_ref18) {
        var from = _ref18.from,
            to = _ref18.to,
            count = _ref18.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " z ").concat(count);
      },
      nextIconButtonText: 'Następna strona'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        var pluralForm = 'gwiazdek';
        var lastDigit = value % 10;

        if ((value < 10 || value > 20) && lastDigit > 1 && lastDigit < 5) {
          pluralForm = 'gwiazdki';
        } else if (value === 1) {
          pluralForm = 'gwiazdka';
        }

        return "".concat(value, " ").concat(pluralForm);
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Wyczyść',
      closeText: 'Zamknij',
      loadingText: 'Ładowanie…',
      noOptionsText: 'Brak opcji',
      openText: 'Otwórz'
    },
    MuiAlert: {
      closeText: 'Zamknij'
    }
  }
};
export var ptBR = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Página anterior',
      labelRowsPerPage: 'Linhas por página:',
      labelDisplayedRows: function labelDisplayedRows(_ref19) {
        var from = _ref19.from,
            to = _ref19.to,
            count = _ref19.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " de ").concat(count);
      },
      nextIconButtonText: 'Próxima página'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Estrela").concat(value !== 1 ? 's' : '');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Limpar',
      closeText: 'Fechar',
      loadingText: 'Carregando…',
      noOptionsText: 'Sem opções',
      openText: 'Abrir'
    },
    MuiAlert: {
      closeText: 'Fechar'
    }
  }
};
export var ptPT = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Página anterior',
      labelRowsPerPage: 'Linhas por página:',
      labelDisplayedRows: function labelDisplayedRows(_ref20) {
        var from = _ref20.from,
            to = _ref20.to,
            count = _ref20.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " de ").concat(count);
      },
      nextIconButtonText: 'Próxima página'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Estrela").concat(value !== 1 ? 's' : '');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Limpar',
      closeText: 'Fechar',
      loadingText: 'A carregar…',
      noOptionsText: 'Sem opções',
      openText: 'Abrir'
    }
  }
};
export var roRO = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Pagina precedentă',
      labelRowsPerPage: 'Rânduri pe pagină:',
      labelDisplayedRows: function labelDisplayedRows(_ref21) {
        var from = _ref21.from,
            to = _ref21.to,
            count = _ref21.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " din ").concat(count);
      },
      nextIconButtonText: 'Pagina următoare'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " St").concat(value !== 1 ? 'ele' : 'ea');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Șterge',
      closeText: 'Închide',
      loadingText: 'Se încarcă…',
      noOptionsText: 'Nicio opțiune',
      openText: 'Deschide'
    },
    MuiAlert: {
      closeText: 'Închide'
    }
  }
};
export var ruRU = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Предыдущая страница',
      labelRowsPerPage: 'Строк на странице:',
      labelDisplayedRows: function labelDisplayedRows(_ref22) {
        var from = _ref22.from,
            to = _ref22.to,
            count = _ref22.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " \u0438\u0437 ").concat(count);
      },
      nextIconButtonText: 'Следующая страница'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        var pluralForm = 'Звёзд';
        var lastDigit = value % 10;

        if (lastDigit > 1 && lastDigit < 5) {
          pluralForm = 'Звезды';
        } else if (lastDigit === 1) {
          pluralForm = 'Звезда';
        }

        return "".concat(value, " ").concat(pluralForm);
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Очистить',
      closeText: 'Закрыть',
      loadingText: 'Загрузка…',
      noOptionsText: 'Нет доступных вариантов',
      openText: 'Открыть'
    },
    MuiAlert: {
      closeText: 'Закрыть'
    }
  }
};
export var skSK = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Predchádzajúca stránka',
      labelRowsPerPage: 'Riadkov na stránke:',
      labelDisplayedRows: function labelDisplayedRows(_ref23) {
        var from = _ref23.from,
            to = _ref23.to,
            count = _ref23.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " z ").concat(count);
      },
      nextIconButtonText: 'Ďalšia stránka'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        if (value === 1) {
          return "".concat(value, " hviezdi\u010Dka");
        }

        if (value >= 2 && value <= 4) {
          return "".concat(value, " hviezdi\u010Dky");
        }

        return "".concat(value, " hviezdi\u010Diek");
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Vymazať',
      closeText: 'Zavrieť',
      loadingText: 'Načítanie…',
      noOptionsText: 'Žiadne možnosti',
      openText: 'Otvoriť'
    },
    MuiAlert: {
      closeText: 'Zavrieť'
    }
  }
};
export var svSE = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Föregående sida',
      labelRowsPerPage: 'Rader per sida:',
      labelDisplayedRows: function labelDisplayedRows(_ref24) {
        var from = _ref24.from,
            to = _ref24.to,
            count = _ref24.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " av ").concat(count);
      },
      nextIconButtonText: 'Nästa sida'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " ").concat(value !== 1 ? 'Stjärnor' : 'Stjärna');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Rensa',
      closeText: 'Stäng',
      loadingText: 'Laddar…',
      noOptionsText: 'Inga alternativ',
      openText: 'Öppen'
    },
    MuiAlert: {
      closeText: 'Stäng'
    }
  }
};
export var trTR = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Önceki sayfa',
      labelRowsPerPage: 'Sayfa başına satır:',
      labelDisplayedRows: function labelDisplayedRows(_ref25) {
        var from = _ref25.from,
            to = _ref25.to,
            count = _ref25.count;
        return "".concat(count, " tanesinden ").concat(from, "-").concat(to === -1 ? count : to);
      },
      nextIconButtonText: 'Sonraki sayfa'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " Y\u0131ld\u0131z");
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Temizle',
      closeText: 'Kapat',
      loadingText: 'Yükleniyor…',
      noOptionsText: 'Seçenek yok',
      openText: 'Aç'
    },
    MuiAlert: {
      closeText: 'Kapat'
    }
  }
};
export var ukUA = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Попередня сторінка',
      labelRowsPerPage: 'Рядків на сторінці:',
      labelDisplayedRows: function labelDisplayedRows(_ref26) {
        var from = _ref26.from,
            to = _ref26.to,
            count = _ref26.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " \u0437 ").concat(count);
      },
      nextIconButtonText: 'Наступна сторінка'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        var pluralForm = 'Зірок';
        var lastDigit = value % 10;

        if (lastDigit > 1 && lastDigit < 5) {
          pluralForm = 'Зірки';
        } else if (lastDigit === 1) {
          pluralForm = 'Зірка';
        }

        return "".concat(value, " ").concat(pluralForm);
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: 'Очистити',
      closeText: 'Згорнути',
      loadingText: 'Завантаження…',
      noOptionsText: 'Немає варіантів',
      openText: 'Розгорнути'
    },
    MuiAlert: {
      closeText: 'Згорнути'
    }
  }
};
export var viVN = {
  props: {
    MuiTablePagination: {
      backIconButtonText: 'Trang trước',
      labelRowsPerPage: 'Số hàng mỗi trang:',
      labelDisplayedRows: function labelDisplayedRows(_ref27) {
        var from = _ref27.from,
            to = _ref27.to,
            count = _ref27.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " trong ").concat(count);
      },
      nextIconButtonText: 'Trang sau'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " sao");
      },
      emptyLabelText: 'Trống'
    },
    MuiAutocomplete: {
      clearText: 'Xóa',
      closeText: 'Đóng',
      loadingText: 'Đang tải…',
      noOptionsText: 'Không có lựa chọn',
      openText: 'Mở'
    },
    MuiAlert: {
      closeText: 'Đóng'
    }
  }
};
export var zhCN = {
  props: {
    MuiTablePagination: {
      backIconButtonText: '上一页',
      labelRowsPerPage: '每页行数:',
      labelDisplayedRows: function labelDisplayedRows(_ref28) {
        var from = _ref28.from,
            to = _ref28.to,
            count = _ref28.count;
        return "".concat(from, "-").concat(to === -1 ? count : to, " \u7684 ").concat(count);
      },
      nextIconButtonText: '下一页'
    },
    MuiRating: {
      getLabelText: function getLabelText(value) {
        return "".concat(value, " \u661F").concat(value !== 1 ? '星' : '');
      },
      emptyLabelText: 'Empty'
    },
    MuiAutocomplete: {
      clearText: '明确',
      closeText: '关',
      loadingText: '载入中…',
      noOptionsText: '没有选择',
      openText: '打开'
    },
    MuiAlert: {
      closeText: '关'
    }
  }
};