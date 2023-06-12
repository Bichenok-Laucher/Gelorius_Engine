function scroll_to_word(){
	pos = $('#text .selectHighlight').position()
	$('#content').jqxPanel('scrollTo', 0, pos.top - 5);
}	

var search_number = 0; //индекс конкретного сочетания из найденных
var search_count = 0;	//количество найденных сочетаний			

//search - поиск слова по нажатию на кнопку "search_button"
$('#search_button').click(function() {
    $('#text').removeHighlight();
    txt = $('#search_text').val();
    if (txt == '')
        return;
    $('#text').highlight(txt);
    search_count = $('#text span.highlight').size() - 1;
    search_number = 0;
    $('#text').selectHighlight(search_number); //выделяем первое слово из найденных
    scroll_to_word(); //скролим страничку к выделяемому слову
});

//clear - очистка выделения по нажатию на кнопку "clear_button"
$('#clear_button').click(function() {
    $('#text').removeHighlight();
});

//prev_search - выделяем предыдущие слово из найденных и скролим страничку к нему
$('#prev_search').click(function() {
    if (search_number == 0)
        return;
    $('#text .selectHighlight').removeClass('selectHighlight');
    search_number--;
    $('#text').selectHighlight(search_number);
    scroll_to_word();
});

//next_search - выделяем следующее слово из найденных и скролим страничку к нему
$('#next_search').click(function() {
    if (search_number == search_count)
        return;
    $('#text .selectHighlight').removeClass('selectHighlight');
    search_number++;
    $('#text').selectHighlight(search_number);
    scroll_to_word();
});
