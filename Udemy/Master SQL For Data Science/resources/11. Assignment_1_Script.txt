create table students
(
	student_no integer,
	student_name varchar(20),
	age integer
);

insert into students values (1, 'Michael', 19);
insert into students values (2, 'Doug', 18);
insert into students values (3, 'Samantha', 21);
insert into students values (4, 'Pete', 20);
insert into students values (5, 'Ralph', 19);
insert into students values (6, 'Arnold', 22);
insert into students values (7, 'Michael', 19);
insert into students values (8, 'Jack', 19);
insert into students values (9, 'Rand', 17);
insert into students values (10, 'Sylvia', 20);

create table courses
(
	course_no varchar(5),
	course_title varchar(20),
	credits integer
);

insert into courses values ('CS110', 'Pre Calculus', 4);
insert into courses values ('CS180', 'Physics', 4);
insert into courses values ('CS107', 'Intro to Psychology', 3);
insert into courses values ('CS210', 'Art History', 3);
insert into courses values ('CS220', 'US History', 3);

create table student_enrollment
(
	student_no integer,
	course_no varchar(5)
);

insert into student_enrollment values (1, 'CS110');
insert into student_enrollment values (1, 'CS180');
insert into student_enrollment values (1, 'CS210');
insert into student_enrollment values (2, 'CS107');
insert into student_enrollment values (2, 'CS220');
insert into student_enrollment values (3, 'CS110');
insert into student_enrollment values (3, 'CS180');
insert into student_enrollment values (4, 'CS220');
insert into student_enrollment values (5, 'CS110');
insert into student_enrollment values (5, 'CS180');
insert into student_enrollment values (5, 'CS210');
insert into student_enrollment values (5, 'CS220');
insert into student_enrollment values (6, 'CS110');
insert into student_enrollment values (7, 'CS110');
insert into student_enrollment values (7, 'CS210');


create table professors
(
	last_name varchar(20),
	department varchar(12),
	salary integer,
	hire_date date
);

insert into professors values ('Chong', 'Science', 88000, '2006-04-18');
insert into professors values ('Brown', 'Math', 97000, '2002-08-22');
insert into professors values ('Jones', 'History', 67000, '2009-11-17');
insert into professors values ('Wilson', 'Astronomy', 110000, '2005-01-15');
insert into professors values ('Miller', 'Agriculture', 82000, '2008-05-08');
insert into professors values ('Williams', 'Law', 105000, '2001-06-05');

create table teach
(
	last_name varchar(20),
	course_no varchar(5)
);

insert into teach values ('Chong', 'CS180');
insert into teach values ('Brown', 'CS110');
insert into teach values ('Brown', 'CS180');
insert into teach values ('Jones', 'CS210');
insert into teach values ('Jones', 'CS220');
insert into teach values ('Wilson', 'CS110');
insert into teach values ('Wilson', 'CS180');
insert into teach values ('Williams', 'CS107');